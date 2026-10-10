#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fly_connectome_chess.py
=======================
Fruit Fly Connectome Chess Bot
------------------------------
A single, self-contained script that wires together five components:

  1. neuPrint connectome access (real API with live-to-mock failover)
  2. python-chess game state + sensory encoding of the 64 squares
  3. A simulated dopamine / Hebbian reinforcement loop over synapse weights
  4. A live 3D wireframe fly rendered with pure pygame (Vector3 + painter's
     algorithm), whose limbs fire in response to real motor-neuron activations
  5. An engine loop where the fly plays vs. a random opponent (or itself),
     evolving its connectome weights in real time.

BIOLOGICAL -> COMPUTATIONAL MAPPING (high level)
------------------------------------------------
  Optic lobe / medulla (ME) neurons .......... 64 "sensory columns", one per
                                               board square (a8..h1). Their
                                               activation vector encodes the
                                               piece occupying the square.
  Lateral horn / wedge / noduli .............. hidden "integration" layers that
                                               pool sensory evidence.
  Descending neurons / VNC motoneurons ....... 192 motor output channels
                                               (64 lift = from-square,
                                                64 place = to-square,
                                                32 wing, 32 abdominal).
  mushroom_body / mushroom_body_lobe ......... plastic synaptic weights, one
                                               matrix per layer, shaped by
                                               dopamine-modulated Hebbian
                                               plasticity.

Dependencies:
    pip install "neuprint-python>=0.4.5" python-chess pygame numpy

Run:
    python fly_connectome_chess.py            # PLAY: you are White, click the
                                              # board on the right half
    python fly_connectome_chess.py --watch    # autopilot: fly vs random
    python fly_connectome_chess.py --selfplay # connectome vs itself
    python fly_connectome_chess.py --offline  # skip the neuPrint API (mock)

    - Left half: 3D wireframe fly reacting to its motor neurons.
    - Right half: live chessboard; click a white piece then a target square.
    - ESC/Q quits, SPACE pauses (or starts the next game after game over),
      I restarts the game, D dumps connectome JSON.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import queue
from datetime import datetime
from collections import deque

import chess
import chess.pgn
import chess.engine

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

APP_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
           else os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
REPLAY_DIR = os.path.join(APP_DIR, "replays")
WEIGHTS_PATH = os.path.join(APP_DIR, "connectome_weights.npz")
DEFAULT_SETTINGS = {
    "difficulty": 6,
    "eco_mode": False,
    "realistic_mode": False,
    "animation_seconds": 0.24,
    "training_games": 5,
    "offline": False,
    "stockfish_path": "",
}
STOCKFISH_ANALYSIS_SECONDS = 0.15
STOCKFISH_MATE_SCORE = 100000
STOCKFISH_MOVE_QUALITY_THRESHOLDS = (
    (10, "Best"),
    (30, "Excellent"),
    (100, "Good"),
    (200, "Inaccuracy"),
    (400, "Mistake"),
    (math.inf, "Blunder"),
)
STOCKFISH_MOVE_QUALITY_COLORS = {
    "Best": (81, 218, 177),
    "Excellent": (91, 190, 224),
    "Good": (122, 205, 126),
    "Inaccuracy": (235, 194, 91),
    "Mistake": (241, 145, 76),
    "Blunder": (234, 93, 106),
}


def classify_stockfish_move(centipawn_loss, best_move="", played_move=""):
    """Return a chess.com-style label and color for a move's eval loss."""
    if best_move and best_move == played_move:
        label = "Best"
        return label, STOCKFISH_MOVE_QUALITY_COLORS[label]
    try:
        loss = float(centipawn_loss)
    except (TypeError, ValueError, OverflowError):
        loss = math.inf
    if not math.isfinite(loss):
        loss = math.inf
    else:
        loss = max(0.0, loss)
    for threshold, label in STOCKFISH_MOVE_QUALITY_THRESHOLDS:
        if loss <= threshold:
            return label, STOCKFISH_MOVE_QUALITY_COLORS[label]
    label = "Blunder"
    return label, STOCKFISH_MOVE_QUALITY_COLORS[label]


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as stream:
            saved = json.load(stream)
        if isinstance(saved, dict):
            settings.update({key: value for key, value in saved.items()
                             if key in DEFAULT_SETTINGS})
    except (OSError, ValueError, TypeError):
        pass

    try:
        settings["difficulty"] = max(1, min(10, int(settings["difficulty"])))
    except (TypeError, ValueError, OverflowError):
        settings["difficulty"] = DEFAULT_SETTINGS["difficulty"]
    try:
        settings["training_games"] = max(
            1, min(50, int(settings["training_games"])))
    except (TypeError, ValueError, OverflowError):
        settings["training_games"] = DEFAULT_SETTINGS["training_games"]
    try:
        duration = float(settings["animation_seconds"])
        settings["animation_seconds"] = (max(0.08, min(0.8, duration))
                                          if math.isfinite(duration) else
                                          DEFAULT_SETTINGS["animation_seconds"])
    except (TypeError, ValueError, OverflowError):
        settings["animation_seconds"] = DEFAULT_SETTINGS["animation_seconds"]
    for key in ("eco_mode", "realistic_mode", "offline"):
        settings[key] = settings[key] is True
    engine_path = settings.get("stockfish_path", "")
    settings["stockfish_path"] = (engine_path.strip()
                                  if isinstance(engine_path, str) else "")
    return settings


def save_settings(settings):
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        temp_path = SETTINGS_PATH + ".tmp"
        data = dict(DEFAULT_SETTINGS)
        data.update({key: settings[key] for key in DEFAULT_SETTINGS if key in settings})
        with open(temp_path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
        os.replace(temp_path, SETTINGS_PATH)
    except OSError as exc:
        print("[settings] could not save preferences: %s" % exc)


def resolve_stockfish_path(configured_path=""):
    """Resolve the configured Stockfish executable or one available on PATH."""
    configured_path = os.path.expandvars(os.path.expanduser(
        str(configured_path or "").strip().strip('"')))
    if configured_path:
        if os.path.isfile(configured_path):
            return os.path.abspath(configured_path)
        located = shutil.which(configured_path)
        if located:
            return located
    return shutil.which("stockfish") or shutil.which("stockfish.exe")


def difficulty_think_seconds(level):
    """Map the visible 1-10 menu difficulty to the bot's search budget."""
    level = max(1, min(10, int(level)))
    return round(0.15 + 8.0 * (level / 10.0) ** 2, 2)


def save_pgn_replay(board, game_no, mode, fly_score):
    """Save a completed or interrupted game as a portable PGN review."""
    os.makedirs(REPLAY_DIR, exist_ok=True)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Fruit Fly Connectome Chess"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_no)
    if mode == "play":
        game.headers["White"] = "Human"
        game.headers["Black"] = "Connectome Fly"
    elif mode == "selfplay":
        game.headers["White"] = "Connectome Fly"
        game.headers["Black"] = "Connectome Fly"
    else:
        game.headers["White"] = "Random Opponent"
        game.headers["Black"] = "Connectome Fly"
    game.headers["Mode"] = mode
    game.headers["FlyScore"] = "%.2f" % fly_score
    outcome = board.outcome(claim_draw=True)
    game.headers["Result"] = outcome.result() if outcome else "*"
    game.headers["Termination"] = (str(outcome.termination) if outcome
                                    else "interrupted or ply limit")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(REPLAY_DIR, "game_%s_%02d.pgn" % (stamp, game_no))
    with open(path, "w", encoding="utf-8") as stream:
        print(game, file=stream, end="\n\n")
    return path


def list_replays():
    if not os.path.isdir(REPLAY_DIR):
        return []
    return sorted((os.path.join(REPLAY_DIR, name)
                   for name in os.listdir(REPLAY_DIR)
                   if name.lower().endswith(".pgn")), reverse=True)

# ============================================================================
# 1. API INITIALIZATION & NEUPRINT SETUP
# ============================================================================
# neuPrint hosts the electron-microscopy connectome of (among others) the adult
# fruit fly CNS.  We authenticate with a token and query real neuron/synapse
# data.  If the network, dataset, or token is unavailable we transparently fall
# back to a structurally faithful MOCK connectome so the script always runs.

NEUPRINT_SERVER = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
# Set NEUPRINT_TOKEN in your environment to opt into neuPrint API access.
# Never commit API tokens; without one, the app can fall back to its mock data.
NEUPRINT_TOKEN = os.environ.get("NEUPRINT_TOKEN", "")

# Region names actually present in the male-cns:v1.0 dataset.
SENSORY_REGION = "ME(R)"          # right optic lobe / medulla
MOTOR_REGION = "VNC"              # ventral nerve cord (DN/motor territory)

# ---------------------------------------------------------------------------
# The 64 chess squares mapped onto sensory columns of the medulla.
# In the real ME, columns are arranged retinotopically; we impose an 8x8
# retinotopy on the board so every square has a dedicated column of neurons.
# ---------------------------------------------------------------------------
FILES = "abcdefgh"


def build_square_ids():
    """Return (square_ids, square_to_idx) where square_ids[idx] == chess square."""
    square_ids, square_to_idx = [], {}
    idx = 0
    for rank in range(7, -1, -1):        # rank 8 down to rank 1 (a8 first)
        for file in range(8):
            sq = chess.square(file, rank)
            square_ids.append(sq)
            square_to_idx[sq] = idx
            idx += 1
    return square_ids, square_to_idx


SQUARE_IDS, SQUARE_TO_IDX = build_square_ids()
N_SQUARES = 64

# Moves are encoded as from-square x to-square -> 64 * 64 potential codes, but
# a chess move only needs a *from* channel and a *to* channel on the motor
# side.  Motor layout (192 channels):
#     [0:64]    'lift'  descending neurons  -> from-square
#     [64:128]  'place' descending neurons  -> to-square
#     [128:160] 'VNC-T' thoracic motoneurons -> wing power
#     [160:192] 'VNC-A' abdominal motoneurons -> turn/abdomen bend
N_MOTOR = 192

# ---------------------------------------------------------------------------
# Mock connectome: region -> list of neuron names.  Real queries use the same
# names where possible so the two paths are interchangeable.
# ---------------------------------------------------------------------------
MOCK_NEURONS = {
    SENSORY_REGION: ["ME(R) column %s%d" % (FILES[f], r + 1)
                     for r in range(8) for f in range(8)],          # 64
    "LO(R)": ["LO(R) %02d" % i for i in range(1, 17)],              # 16
    "LOP(R)": ["LOP(R) %02d" % i for i in range(1, 9)],             # 8
    "AOTU(R)": ["AOTU(R) %02d" % i for i in range(1, 5)],           # 4
    "WED(R)": ["WED(R) %02d" % i for i in range(1, 7)],             # 6
    "SIP(R)": ["SIP(R) %02d" % i for i in range(1, 7)],             # 6
    "SLP(R)": ["SLP(R) %02d" % i for i in range(1, 7)],             # 6
    "SCL(R)": ["SCL(R) %02d" % i for i in range(1, 5)],             # 4
    "NO1(R)": ["NO1(R) %02d" % i for i in range(1, 5)],             # 4
    "MB(R)": ["MB(R) KC %02d" % i for i in range(1, 9)],            # 8
    "CRE(R)": ["CRE(R) %02d" % i for i in range(1, 5)],             # 4
    "INP": ["INP %02d" % i for i in range(1, 5)],                   # 4
    "LAL(R)": ["LAL(R) %02d" % i for i in range(1, 7)],             # 6
    "PVLP(R)": ["PVLP(R) %02d" % i for i in range(1, 5)],           # 4
    "GNG": ["GNG %02d" % i for i in range(1, 5)],                   # 4
    MOTOR_REGION: (["DNp%02d" % i for i in range(1, 65)]              # 64 lift DNs
                   + ["DNb%02d" % i for i in range(1, 65)]            # 64 place DNs
                   + ["VNC-T%02d" % i for i in range(1, 33)]          # 32 thoracic
                   + ["VNC-A%02d" % i for i in range(1, 33)]),        # 32 abdominal
}

# Signal path (sensory -> ... -> motor).  Plastic weights live on the edges.
CONNECTOME_PATH = [
    SENSORY_REGION, "LO(R)", "LOP(R)", "AOTU(R)", "WED(R)", "SIP(R)",
    "SLP(R)", "SCL(R)", "NO1(R)", "MB(R)", "CRE(R)", "INP", "LAL(R)",
    "PVLP(R)", "GNG", MOTOR_REGION,
]


class MockClient(object):
    """A tiny offline stand-in for the neuprint Client.

    Generates a plausible feed-forward connectome between the regions in
    CONNECTOME_PATH with realistic-looking neuron names, so every downstream
    component (propagation, learning, visualization) works identically whether
    the live neuPrint API or this mock supplied the graph.
    """

    def __init__(self):
        self.neurons = {}
        for region, names in MOCK_NEURONS.items():
            for name in names:
                bodyId = abs(hash((region, name))) % 1_000_000
                self.neurons[bodyId] = {"bodyId": bodyId, "name": name,
                                        "instance": name, "rois": [region]}
        self.edges = {}
        for a, b in zip(CONNECTOME_PATH[:-1], CONNECTOME_PATH[1:]):
            src, dst = [], []
            for bodyId, meta in self.neurons.items():
                if a in meta["rois"]:
                    src.append(bodyId)
                if b in meta["rois"]:
                    dst.append(bodyId)
            connections = []
            rng = random.Random(hash((a, b)) & 0xFFFFFFFF)
            for s in src:
                for d in rng.sample(dst, k=min(len(dst), 8)):
                    connections.append((s, d, rng.randint(5, 40)))
            self.edges[(a, b)] = connections
        self._by_roi = None

    def fetch_custom(self, query, *, debug=False):
        """Minimal fetch_custom used by fetch_connectome_graph."""
        q = query if isinstance(query, str) else " ".join(query)
        self._by_roi = self._by_roi or {}
        key = None
        for region in CONNECTOME_PATH:
            if region in q:
                key = region
                break
        rows = []
        if key and key in self.edges:
            for s, d, w in self.edges[key]:
                rows.append({"bodyId": s, "partner": d, "weight": w})
        return rows

    def fetch_neurons(self, regex, rois=None):
        out = {}
        low = regex.lower()
        for bodyId, meta in self.neurons.items():
            hay = (meta["name"] + " " + meta["instance"]).lower()
            if low in hay:
                out[bodyId] = meta
        return out

    def fetch_shortest_paths(self, upstream_bodyId, downstream_bodyId,
                             min_weight=5):
        """Very small BFS over the mock edges for path queries."""
        adj = {}
        for (a, b), conns in self.edges.items():
            for s, d, w in conns:
                if w >= min_weight:
                    adj.setdefault(s, []).append(d)
        # BFS from upstream towards downstream (bounded)
        from collections import deque as _dq
        parents = {upstream_bodyId: None}
        q = _dq([upstream_bodyId])
        while q:
            cur = q.popleft()
            if cur == downstream_bodyId:
                break
            for nxt in adj.get(cur, []):
                if nxt not in parents:
                    parents[nxt] = cur
                    q.append(nxt)
        if downstream_bodyId not in parents:
            return []
        path, node = [], downstream_bodyId
        while node is not None:
            path.append(node)
            node = parents[node]
        return [list(reversed(path))]


class ConnectomeGateway(object):
    """Owns the neuPrint client (or mock) and exposes a uniform graph API."""

    def __init__(self, token, server, dataset, offline=False, verbose=True):
        self.client = None
        self.live = False
        self.verbose = verbose
        if not offline:
            self._try_live(token, server, dataset)
        if not self.live:
            if verbose:
                print("[neuPrint] using MOCK connectome (structural stand-in).")
            self.client = MockClient()

    def _try_live(self, token, server, dataset):
        try:
            from neuprint import Client  # neuprint-python
            self.client = Client(server, dataset=dataset, token=token)
            # Verify credentials with a cheap real query (1 neuron).
            probe = self.client.fetch_custom(
                "MATCH (n:Neuron) RETURN n.bodyId AS bodyId LIMIT 1")
            _ = self._records_from_fetch(probe)
            self.live = True
            if self.verbose:
                print("[neuPrint] LIVE client -> %s  dataset=%s" % (server, dataset))
        except Exception as exc:
            if self.verbose:
                print("[neuPrint] live client unavailable (%s); "
                      "falling back to mock." % exc)
            self.live = False

    @staticmethod
    def _records_from_fetch(rows):
        """Normalise neuPrint results (pandas DataFrame / list) to dicts."""
        if rows is None:
            return []
        to_dict = getattr(rows, "to_dict", None)
        if to_dict is not None:
            try:
                return rows.to_dict("records")
            except Exception:
                return []
        return list(rows)

    @staticmethod
    def _num(rec, *keys):
        """Pull the first present key out of a result record as an int."""
        for k in keys:
            if k in rec:
                try:
                    return int(rec[k])
                except Exception:
                    continue
        return 0

    @staticmethod
    def _pool_items(region):
        return [(hash(p) % 1_000_000, {"name": p, "instance": p})
                for p in MOCK_NEURONS[region]]

    # -- graph construction -------------------------------------------------
    def fetch_sensory_neurons(self, region=SENSORY_REGION, n=N_SQUARES):
        """Map the 64 board squares onto medulla sensory neuron names."""
        items = []
        if self.live:
            try:
                result = self.client.fetch_neurons(r"ME\(R\)", rois=[region])
                items = self._records_from_fetch(result)
                items = [(int(r["bodyId"]), {"name": str(r.get("instance", "ME?")),
                                              "instance": str(r.get("instance", "ME?"))})
                         for r in items if "bodyId" in r]
            except Exception:
                items = []
        if len(items) < n:
            items += self._pool_items(region)      # pad so indexing is safe
        items = items[:n]
        return [{"bodyId": b, "name": m.get("instance", m.get("name", "ME?"))}
                for b, m in items]

    def fetch_motor_neurons(self, region=MOTOR_REGION, n=N_MOTOR):
        """Fetch descending-neuron / VNC motoneuron channels for move output."""
        items = []
        if self.live:
            try:
                result = self.client.fetch_neurons(r"DN|VNC", rois=[region])
                items = self._records_from_fetch(result)
                items = [(int(r["bodyId"]), {"name": str(r.get("instance", "VNC?")),
                                              "instance": str(r.get("instance", "VNC?"))})
                         for r in items if "bodyId" in r]
            except Exception:
                items = []
        if len(items) < n:
            items += self._pool_items(region)
        items = items[:n]
        return [{"bodyId": b, "name": m.get("instance", m.get("name", "VNC?"))}
                for b, m in items]

    def fetch_connectome_graph(self):
        """Return {region_pair: [(pre_bodyId, post_bodyId, weight), ...]}.

        Live mode issues a real Cypher query per region pair via
        ``client.fetch_custom``; the mock answers from its own edge table.
        """
        graph = {}
        for a, b in zip(CONNECTOME_PATH[:-1], CONNECTOME_PATH[1:]):
            query = ("MATCH (a:`%s`)-[w]->(b:`%s`) "
                     "RETURN a.bodyId AS bodyId, b.bodyId AS partner, "
                     "w.weight AS weight LIMIT 5000" % (a, b))
            records = self._records_from_fetch(self.client.fetch_custom(query))
            conns = []
            for r in records:
                s = self._num(r, "bodyId", "a.bodyId", ":START_ID")
                d = self._num(r, "partner", "b.bodyId", ":END_ID")
                w = self._num(r, "weight", "w.weight", "weight:TYPE") or 1
                if s and d:
                    conns.append((s, d, w))
            graph[(a, b)] = conns
        return graph


# ============================================================================
# 2. CHESS LOGIC INTEGRATION  +  CONNECTOME SIGNAL PROPAGATION
# ============================================================================

class ConnectomeNetwork(object):
    """Feed-forward signal propagation over the fetched connectome graph.

    Layers correspond to real neuPrint ROIs; the neuron *names* come from the
    API (live) or the mock.  Synaptic weights start at the graph's anatomical
    strengths and are then modulated by the dopamine learning loop.
    """

    def __init__(self, gateway):
        self.gateway = gateway
        self.sensory = gateway.fetch_sensory_neurons()
        self.motor = gateway.fetch_motor_neurons()
        self.graph = gateway.fetch_connectome_graph()

        # sensory layer is the input, motor layer is the output
        self.layers = [SENSORY_REGION] + CONNECTOME_PATH[1:]
        self.regions = list(self.layers)

        # ---- build weight matrices between consecutive regions -------------
        self.weights = {}
        self.wmax = {}
        for (a, b) in zip(self.regions[:-1], self.regions[1:]):
            src_ids = [m["bodyId"] for m in self.layer_neurons(a)]
            dst_ids = [m["bodyId"] for m in self.layer_neurons(b)]
            W = np.zeros((len(dst_ids), len(src_ids)), dtype=float)
            connections = self.graph.get((a, b), [])
            for s, d, w in connections:
                if s in src_ids and d in dst_ids:
                    W[dst_ids.index(d), src_ids.index(s)] = float(w)
            # If the API returned nothing usable, seed a sparse random matrix so
            # signals still flow (the mock guarantees coverage, live may not).
            if W.sum() == 0.0:
                rng = random.Random(hash((a, b)) & 0xFFFFFFFF)
                for _ in range(min(len(src_ids) * len(dst_ids) // 4, 4096)):
                    W[rng.randrange(len(dst_ids)), rng.randrange(len(src_ids))] = \
                        float(rng.randint(5, 40))
            self.weights[(a, b)] = W
            self.wmax[(a, b)] = max(W.max(), 1.0)

        # index maps for O(1) lookup during play / rendering
        self.sensory_names = [m["name"] for m in self.sensory]
        self.motor_names = [m["name"] for m in self.motor]
        self.sensory_index = {m["bodyId"]: i for i, m in enumerate(self.sensory)}
        self.motor_index = {m["bodyId"]: i for i, m in enumerate(self.motor)}

        # recent activations, for the 3D visualizer and learning
        self.last_sensory = np.zeros(N_SQUARES)
        self.last_hidden = {}
        self.last_motor = np.zeros(N_MOTOR)
        self.last_dopamine = 0.0
        self.last_pathway = (0, 0, 0.0)
        self.da_history = []            # (reward, dopamine) pairs, last 200

    def layer_neurons(self, region):
        if region == SENSORY_REGION:
            return self.sensory
        if region == MOTOR_REGION:
            return self.motor
        pool = MOCK_NEURONS[region]
        return [{"bodyId": hash(p) % 1_000_000, "name": p} for p in pool]

    def mean_weight(self):
        """Mean |W| across all plastic layers (learning progress indicator)."""
        vals = [float(np.mean(np.abs(W))) for W in self.weights.values()]
        return float(np.mean(vals)) if vals else 0.0

    # ---- board -> sensory activation --------------------------------------
    def encode_board(self, board, body=None):
        """Encode the board layout into medulla sensory activation.

        BIOLOGY: each visual column of the medulla reports local luminance /
        salience.  COMPUTATION: column i fires with an amplitude proportional
        to the material standing on square i, signed by piece colour (own
        pieces excite, enemy pieces excite more weakly but negatively gate).
        """
        x = np.zeros(N_SQUARES)
        for sq, piece in board.piece_map().items():
            idx = SQUARE_TO_IDX[sq]
            val = PIECE_SIGNAL[piece.piece_type]
            x[idx] = val if piece.color == chess.WHITE else -0.6 * val
        # SENSORIMOTOR FEEDBACK: the walking fly senses the board from its own
        # body position - squares near its feet get salience (proximity
        # attention), so where it has walked changes what it perceives next.
        if body is not None:
            for i, sq in enumerate(SQUARE_IDS):
                wx = (chess.square_file(sq) - 3.5) * 4.0
                wz = (3.5 - chess.square_rank(sq)) * 4.0
                d2 = (wx - body.pos.x) ** 2 + (wz - body.pos.z) ** 2
                x[i] += 0.5 * math.exp(-d2 / 24.0)
        # normalise into [ -1, 1 ]
        m = np.max(np.abs(x)) or 1.0
        self.last_sensory = x / m
        return self.last_sensory

    # ---- forward pass ------------------------------------------------------
    def propagate(self, sensory):
        """Push the sensory vector through the connectome to the motor layer."""
        acts = {self.regions[0]: sensory}
        hidden = {}
        for a, b in zip(self.regions[:-1], self.regions[1:]):
            W = self.weights[(a, b)]
            prev = acts[a]
            z = W @ prev if W.shape[1] == prev.shape[0] else W @ (prev[:W.shape[1]])
            # tanh-like graded response keeps activations bounded, mimicking
            # saturating synapses.
            z = np.tanh(z / max(1.0, self.wmax[(a, b)] * 0.5))
            acts[b] = z
            if b != MOTOR_REGION:
                hidden[b] = z
        self.last_hidden = hidden
        self.last_motor = acts[MOTOR_REGION]
        return acts

    # ---- motor layer -> legal chess move -----------------------------------
    def motor_to_move(self, board, acts_motor):
        """Pick a legal move from motor neuron activations.

        BIOLOGY: descending neurons (DNxx) tonically bias wing/leg motor
        programs; the most co-active lift+place pair wins.  COMPUTATION: the
        from-square is the argmax over 'lift' channels (0..63) and the
        to-square the argmax over 'place' channels (64..127); if that pair is
        not a legal move we take the best *legal* move that maximises
        lift + place drive.
        """
        lift = np.asarray(acts_motor[:64], dtype=float)
        place = np.asarray(acts_motor[64:128], dtype=float)
        legal = list(board.legal_moves)
        if not legal:
            return None
        scored = []
        for mv in legal:
            fi = SQUARE_TO_IDX[mv.from_square]
            ti = SQUARE_TO_IDX[mv.to_square]
            scored.append((lift[fi] + place[ti], lift[fi], place[ti], mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0]

    # ---- dopamine / Hebbian plasticity -------------------------------------
    def apply_dopamine(self, pathway_score, from_idx, to_idx, reward):
        """Three-factor Hebbian rule with dopamine as the neuromodulator.

        dw = eta * dopamine * pre  (x)  post

        - reward > 0  -> dopamine flood -> potentiate active pathway
        - reward < 0  -> synaptic depression of the pathway that just fired
        - reward == 0 -> passive decay towards anatomical baseline
        """
        dopamine = DA_GAIN * reward
        for a, b in zip(self.regions[:-1], self.regions[1:]):
            W = self.weights[(a, b)]
            pre = self.last_hidden.get(a, self.last_sensory if a == SENSORY_REGION
                                       else None)
            if pre is None:
                continue
            post = self.last_hidden.get(b, self.last_motor if b == MOTOR_REGION
                                        else None)
            if post is None:
                continue
            n_pre, n_post = min(W.shape[1], pre.shape[0]), min(W.shape[0],
                                                               post.shape[0])
            if n_pre == 0 or n_post == 0:
                continue
            hebb = np.outer(post[:n_post], pre[:n_pre])
            W += LEARNING_RATE * dopamine * hebb
            # soft clamp to keep the network from exploding or dying
            np.clip(W, -self.wmax[(a, b)] * 4.0, self.wmax[(a, b)] * 4.0,
                    out=W)
            self.weights[(a, b)] = W
        self.last_dopamine = dopamine
        self.last_pathway = (from_idx, to_idx, pathway_score)
        self.da_history.append((float(reward), float(dopamine)))
        if len(self.da_history) > 200:
            self.da_history.pop(0)
        return dopamine

    def save_weights(self, path="connectome_weights.npz"):
        """Persist learned synapses for training runs and later matches."""
        arrays = {"w_%d" % i: W for i, W in enumerate(self.weights.values())}
        arrays["metadata"] = np.array(json.dumps({
            "regions": self.regions,
            "keys": ["%s->%s" % k for k in self.weights],
            "wmax": {"%s->%s" % k: float(v) for k, v in self.wmax.items()},
        }))
        np.savez_compressed(path, **arrays)
        return os.path.abspath(path)

    def load_weights(self, path):
        """Load only shape-compatible learned matrices; anatomy stays local."""
        try:
            data = np.load(path, allow_pickle=False)
            keys = list(self.weights.keys())
            loaded = 0
            for i, key in enumerate(keys):
                name = "w_%d" % i
                if name in data and data[name].shape == self.weights[key].shape:
                    self.weights[key][...] = data[name]
                    loaded += 1
            data.close()
            return loaded
        except (OSError, ValueError, KeyError):
            return 0

    def dump_connectome(self, path="connectome_dump.json"):
        """Write a JSON snapshot of the evolved connectome (debug / analysis)."""
        snap = {
            "dataset": NEUPRINT_DATASET,
            "live_api": bool(self.gateway.live),
            "regions": self.regions,
            "layer_sizes": {r: len(self.layer_neurons(r)) for r in self.regions},
            "mean_abs_weight": {"%s->%s" % k: float(np.mean(np.abs(W)))
                                for k, W in self.weights.items()},
            "max_abs_weight": {"%s->%s" % k: float(np.max(np.abs(W)))
                               for k, W in self.weights.items()},
            "last_dopamine": float(self.last_dopamine),
            "da_history": self.da_history,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2)
        return os.path.abspath(path)


PIECE_SIGNAL = {
    chess.PAWN: 0.5, chess.KNIGHT: 0.8, chess.BISHOP: 0.8,
    chess.ROOK: 1.0, chess.QUEEN: 1.4, chess.KING: 1.6,
}

# ============================================================================
# 3. SIMULATED DOPAMINE REINFORCEMENT LOOP (PPLab neuron, PPLab-γ1etc.)
# ============================================================================
# In Drosophila the mushroom body output neurons (MBONs) are modulated by
# dopaminergic neurons (PPL1/PAM clusters).  Here the scalar `reward` stands in
# for a dopamine spike:  capture value, check, mate and material swing map to
# reward magnitude; blunders (hanging a piece) map to a negative spike.

LEARNING_RATE = 0.02      # eta in the three-factor rule
DA_GAIN = 2.5             # dopamine gain (scaling of the RPE)
MATE_REWARD = 8.0
CHECK_REWARD = 0.8
CAPTURE_SCALE = 0.35      # reward per point of captured material
BLUNDER_SCALE = 0.45      # penalty per centipawn of material lost
DRAW_REWARD = 0.0
TEMPERATURE = 0.35        # exploration for the softmax over motor drive
THINK_TIME = 5.0          # seconds the fly "deliberates" before each move
MATE_CP = 100000          # centipawn value of checkmate in the search

# --- REALISTIC MODE: the fly-brain hardware budget -------------------------
# A real fly runs its whole CNS on ~100 uW and owns no search hardware.  In
# this mode the animal knows the rules but every move is a sensory-triggered
# reflex chosen in fly-real time (< 300 ms), through the same connectome and
# the same dopamine plasticity as every other mode.
FLY_REACTION_TIME = 0.25  # s of "photoreceptor + VNC latency" per decision
FLY_INSTINCT = {          # innate salience of contact (capture) targets
    chess.PAWN: 2.2, chess.KNIGHT: 4.0, chess.BISHOP: 4.2,
    chess.ROOK: 5.5, chess.QUEEN: 8.0, chess.KING: 9.0,
}
CHECK_PANIC = 1.6         # escape bonus when in check (predator-avoidance)
CHECK_SIGNAL = 0.8        # preference for giving check ("poke the threat")
PST_FOLLOW = 0.012        # weak learned-shaped pull of good squares (cp scale)
CONNECTOME_BIAS_REAL = 0.30  # dopamine-learned connectome drive as soft prior

# Material values in centipawns for the deliberation search.
PIECE_CP = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Piece-square tables (simplified evaluation, white's perspective, a8..h1).
# These give the fly positional common sense: central pawns/knights, castled
# king, active rooks... so it stops playing random-looking nonsense.
PST = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0,
                 50, 50, 50, 50, 50, 50, 50, 50,
                 10, 10, 20, 30, 30, 20, 10, 10,
                 5, 5, 10, 25, 25, 10, 5, 5,
                 0, 0, 0, 20, 20, 0, 0, 0,
                 5, -5, -10, 0, 0, -10, -5, 5,
                 5, 10, 10, -20, -20, 10, 10, 5,
                 0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50,
                   -40, -20, 0, 0, 0, 0, -20, -40,
                   -30, 0, 10, 15, 15, 10, 0, -30,
                   -30, 5, 15, 20, 20, 15, 5, -30,
                   -30, 0, 15, 20, 20, 15, 0, -30,
                   -30, 5, 10, 15, 15, 10, 5, -30,
                   -40, -20, 0, 5, 5, 0, -20, -40,
                   -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20,
                   -10, 0, 0, 0, 0, 0, 0, -10,
                   -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 5, 5, 10, 10, 5, 5, -10,
                   -10, 0, 10, 10, 10, 10, 0, -10,
                   -10, 10, 10, 10, 10, 10, 10, -10,
                   -10, 5, 0, 0, 0, 0, 5, -10,
                   -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 0, 0, 0, 0, 0,
                 5, 10, 10, 10, 10, 10, 10, 5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 0, 0, 0, 5, 5, 0, 0, 0],
    chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 5, 5, 5, 0, -10,
                  -5, 0, 5, 5, 5, 5, 0, -5,
                  0, 0, 5, 5, 5, 5, 0, -5,
                  -10, 5, 5, 5, 5, 5, 0, -10,
                  -10, 0, 5, 0, 0, 0, 0, -10,
                  -20, -10, -10, -5, -5, -10, -10, -20],
    chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -20, -30, -30, -40, -40, -30, -30, -20,
                 -10, -20, -20, -20, -20, -20, -20, -10,
                 20, 20, 0, 0, 0, 0, 20, 20,
                 20, 30, 10, 0, 0, 10, 30, 20],
}


def fly_instinct_move(board, motor):
    """Innate visual->motor reflex for REALISTIC mode: the fly-brain's move.

    HARDWARE BUDGET: a real fly CNS has no search hardware.  This is a
    stimulus-response readout over the same 64-column medulla -> ... -> VNC
    connectome the bot always uses - one pass, no tree, no lookahead - so the
    whole "decision" costs microseconds and fits between wingbeats.

    The instincts are simple sensorimotor taxes (fixed priorities):

      1. CHECK PANIC   - in check, anything that stops the check or captures
                         the checker is picked first (predator-avoidance).
      2. CONTACT DRIVE - an enemy piece on a square we can strike triggers an
                         attack; bigger prey, bigger expected DA burst, so
                         bigger instinct score.
      3. SAFETY        - landing where a more valuable enemy piece attacks is
                         penalised (negative score, not a hard veto: flies
                         are not that clever).
      4. CROWD AHEAD   - weak innate chemotaxis: advance pawns, creep toward
                         the enemy king, give check (poke the big animal).
      5. CONNECTOME    - the dopamine-learned synaptic weights of the pathway
                         bias the choice, exactly as in every other mode.

    Returns [(score, move)] sorted best-first; the score feeds the same
    dopamine three-factor rule as the search-based modes.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []
    us = board.turn
    them = not us
    king_sq = board.king(us)

    scored = []
    for mv in legal:
        s = 0.0
        piece = board.piece_at(mv.from_square)
        pt = piece.piece_type if piece else chess.PAWN

        # ---- 1. predator avoidance: we are in check -----------------------
        if board.is_check():
            board.push(mv)
            safe = not board.is_check()      # anything that stops the check
            board.pop()
            if safe:
                s += CHECK_PANIC
            if board.is_capture(mv) and mv.to_square in board.checkers():
                s += 2.0                     # eat the attacker itself

        # ---- 2. contact drive: prey within reach --------------------------
        if board.is_capture(mv):
            victim = board.piece_at(mv.to_square)
            s += FLY_INSTINCT.get(victim.piece_type, 1.0) if victim else 1.0

        # ---- 3. don't land on a predator's tongue -------------------------
        board.push(mv)
        their_attackers = [a for a in board.attackers(them, mv.to_square)
                           if board.piece_at(a)]
        if their_attackers:
            worst_threat = min(PIECE_CP[board.piece_at(a).piece_type]
                               for a in their_attackers)
            defenders = list(board.attackers(us, mv.to_square))
            # a defended landing square makes a trade tolerable; an undefended
            # one stings - but the fly only "feels" it, it does not calculate
            if not defenders or worst_threat > PIECE_CP[pt] + 50:
                # cap: even the enemy king's menace is just a feeling
                s -= min(worst_threat, 1000.0) / 300.0
        # small bonus for giving check: poke the big animal, it flinches
        if board.is_check():
            s += CHECK_SIGNAL
        # ---- 4. mild chemotaxis toward the enemy king ----------------------
        if king_sq is not None:
            kx, ky = king_sq % 8, king_sq // 8
            fx, fy = mv.from_square % 8, mv.from_square // 8
            tx, ty = mv.to_square % 8, mv.to_square // 8
            if abs(tx - kx) + abs(ty - ky) < abs(fx - kx) + abs(fy - ky):
                s += 0.15                    # closing in on the threat
        board.pop()

        # learned-shaped pull of good squares (PST), very weak
        tbl = PST[pt]
        idx = mv.to_square if us == chess.WHITE else (mv.to_square ^ 56)
        s += PST_FOLLOW * tbl[idx]

        # dopamine-learned connectome drive as a soft prior (shared pathway)
        s += CONNECTOME_BIAS_REAL * float(
            motor[SQUARE_TO_IDX[mv.from_square]]
            + motor[64 + SQUARE_TO_IDX[mv.to_square]])

        scored.append((s, mv))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def evaluate_board(board):
    """Static evaluation in centipawns from the side-to-move's perspective.

    COMPUTATION: material + piece-square tables.  This is what the fly's
    'deliberation' (dopamine-gated recurrent replay, if you like) maximises.
    """
    score = 0.0
    for sq, piece in board.piece_map().items():
        tbl = PST[piece.piece_type]
        idx = sq if piece.color == chess.WHITE else (sq ^ 56)   # mirror rank
        v = PIECE_CP[piece.piece_type] + tbl[idx]
        score += v if piece.color == chess.WHITE else -v
    return score if board.turn == chess.WHITE else -score


class SearchTimeout(Exception):
    """Internal signal used to discard an unfinished iterative-deepening pass."""


def _move_order_score(board, move):
    """Try forcing/capturing moves first so alpha-beta prunes more effectively."""
    score = 0
    if move.promotion:
        score += 20000 + PIECE_CP[move.promotion]
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = (PIECE_CP[chess.PAWN] if board.is_en_passant(move)
                        else PIECE_CP[victim.piece_type] if victim else 0)
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_CP[attacker.piece_type] if attacker else 0
        score += 10000 + victim_value * 10 - attacker_value
    return score


def _ordered_legal_moves(board):
    return sorted(board.legal_moves,
                  key=lambda move: _move_order_score(board, move), reverse=True)


def quiescence(board, alpha, beta, deadline, ply=0, qply=0):
    """Resolve forcing captures/check evasions before applying static evaluation."""
    if time.perf_counter() >= deadline:
        raise SearchTimeout
    if board.is_checkmate():
        return -MATE_CP + ply
    if board.is_game_over(claim_draw=True):
        return 0.0
    if qply >= 12:
        return evaluate_board(board)

    in_check = board.is_check()
    if in_check:
        best = -1e18
        moves = _ordered_legal_moves(board)
    else:
        best = evaluate_board(board)
        if best >= beta:
            return best
        alpha = max(alpha, best)
        moves = [move for move in board.legal_moves
                 if board.is_capture(move) or move.promotion]
        moves.sort(key=lambda move: _move_order_score(board, move),
                   reverse=True)

    for move in moves:
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        board.push(move)
        try:
            value = -quiescence(board, -beta, -alpha, deadline,
                                ply=ply + 1, qply=qply + 1)
        finally:
            board.pop()
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def negamax(board, depth, alpha, beta, deadline, ply=0):
    """Depth-limited negamax; timeout aborts, rather than scores a partial tree."""
    if time.perf_counter() >= deadline:
        raise SearchTimeout
    if board.is_checkmate():
        return -MATE_CP + ply
    if board.is_game_over(claim_draw=True):
        return 0.0
    if depth <= 0:
        return quiescence(board, alpha, beta, deadline, ply=ply)
    best = -1e18
    for move in _ordered_legal_moves(board):
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        board.push(move)
        try:
            value = -negamax(board, depth - 1, -beta, -alpha,
                             deadline, ply=ply + 1)
        finally:
            board.pop()
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def search_root_moves(board, legal_moves, think_seconds, think_cb=None,
                      max_depth=64):
    """Return scores from the deepest *fully completed* search iteration."""
    started = time.perf_counter()
    deadline = started + max(0.0, float(think_seconds))
    ordered = sorted(enumerate(legal_moves),
                     key=lambda item: _move_order_score(board, item[1]),
                     reverse=True)
    completed_scores = None
    completed_depth = 0

    for depth in range(1, max_depth + 1):
        iteration_scores = [None] * len(legal_moves)
        try:
            for index, move in ordered:
                if time.perf_counter() >= deadline:
                    raise SearchTimeout
                board.push(move)
                try:
                    if board.is_checkmate():
                        score = MATE_CP - 1
                    elif board.is_game_over(claim_draw=True):
                        score = 0.0
                    else:
                        score = -negamax(board, depth - 1,
                                         -1e18, 1e18, deadline, ply=1)
                finally:
                    board.pop()
                iteration_scores[index] = score
        except SearchTimeout:
            break

        completed_scores = iteration_scores
        completed_depth = depth
        if think_cb:
            think_cb(time.perf_counter() - started, depth)

    return completed_scores, completed_depth


def move_reward(board, move):
    """Compute the primary reward for a move *before* it is made.

    BIOLOGY: dopamine neurons respond to unexpected appetitive cues
    (capture = food, mate = success) and to predicted losses (shock).
    """
    r = 0.0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        # en-passant captures have an empty target square -> pawn
        vtype = victim.piece_type if victim else chess.PAWN
        r += CAPTURE_SCALE * PIECE_SIGNAL[vtype]
    board.push(move)
    if board.is_checkmate():
        r += MATE_REWARD
    elif board.is_check():
        r += CHECK_REWARD
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome and outcome.winner == chess.WHITE:
            r += MATE_REWARD
        elif outcome and outcome.winner is None:
            r += DRAW_REWARD
    board.pop()
    return r


def material_value(board, color):
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    total = 0
    for piece in board.piece_map().values():
        if piece.color == color:
            total += vals[piece.piece_type]
    return total


def blunder_penalty(board_before, board_after, mover_color):
    """Negative reward when the mover loses material after their move."""
    before = material_value(board_before, mover_color)
    after = material_value(board_after, mover_color)
    return -BLUNDER_SCALE * max(0, before - after)


# ============================================================================
# 4. LIVE 3D PYGAME VISUALIZER
# ============================================================================
# Pure pygame 3D: pygame.math.Vector3 for geometry, manual perspective
# projection, painter's algorithm for depth sorting.  Motor neuron activations
# drive limb rotations (legs = DN lift/place channels, wings = thoracic VNC
# channels for downstroke, abdomen for turn bias).

WHITE = (240, 240, 240)
BLACK = (15, 15, 20)
GREY = (90, 90, 100)
RED = (220, 60, 60)
GREEN = (70, 230, 110)
CYAN = (90, 200, 255)
AMBER = (255, 190, 70)
VIOLET = (180, 110, 255)

SCREEN_W, SCREEN_H = 1920, 1080     # full-HD window
TARGET_FPS = 240                   # render/input target for every mode
FOV = 420.0

# ----------------------------------------------------------------------------
# TRUE SPLIT-PANE LAYOUT: two independent viewports, no overlays.
#   left pane  [0 .. SPLIT_X)      3D wireframe fly + dopamine FX + HUD
#   right pane [SPLIT_X .. 1920)   dedicated 2D chessboard pane
# ----------------------------------------------------------------------------
SPLIT_X = SCREEN_W // 2                     # pane divider (x = 960)
BOARD_SIZE = 800
SQ_PX = BOARD_SIZE // 8                     # 100 px squares
BOARD_X0 = SPLIT_X + (SCREEN_W - SPLIT_X - BOARD_SIZE) // 2
BOARD_Y0 = 110
VIEW3D_CX = SPLIT_X // 2                    # 3D viewport centre (left pane)


def v3(x, y, z):
    return pygame.math.Vector3(x, y, z)


def rot_around(vec, axis, deg):
    """Rotate Vector3 `vec` around a unit axis by `deg` degrees (Rodrigues)."""
    axis = pygame.math.Vector3(axis).normalize()
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (vec * c
            + axis.cross(vec) * s
            + axis * (axis.dot(vec)) * (1.0 - c))


def project(pt, cam):
    """Perspective-project a world Vector3 with a proper look-at camera.

    Builds a forward/right/up basis from the orbit camera's yaw/pitch/target,
    then projects into the LEFT pane of the window (the right pane is the
    dedicated chessboard viewport).
    """
    p = pt - cam.pos
    yr, pr = math.radians(cam.yaw), math.radians(cam.pitch)
    # forward: from camera toward the look-at target (unit length)
    forward = v3(-math.sin(yr) * math.cos(pr),
                 -math.sin(pr),
                 math.cos(yr) * math.cos(pr))
    # right: horizontal, perpendicular to forward (matches original handedness)
    right = v3(math.cos(yr), 0.0, math.sin(yr))
    # up: completes the orthonormal basis
    up = v3(-math.sin(pr) * math.sin(yr),
            math.cos(pr),
            math.sin(pr) * math.cos(yr))
    depth = p.dot(forward)
    if depth < 1.0:
        depth = 1.0
    scale = FOV / depth
    return (int(VIEW3D_CX + p.dot(right) * scale),
            int(SCREEN_H / 2 - p.dot(up) * scale)), depth


class Camera(object):
    """Blender-style orbit (turntable) camera around the fly.

    MIDDLE MOUSE + drag orbits (yaw / pitch, clamped like Blender's viewport);
    mouse wheel zooms.  When the user is not dragging, the camera auto-rotates
    slowly so the fly is always seen from a moving viewpoint.
    """

    def __init__(self):
        self.target = v3(0, 0, 0)       # look-at point (fly centre)
        self.dist = 36.0                # orbit radius
        self.yaw = 25.0                 # horizontal orbit angle (deg)
        self.pitch = 22.0               # elevation angle (deg)
        self.drag_speed = 0.4           # deg per pixel of MMB drag
        self.pitch_min, self.pitch_max = -5.0, 80.0
        self._pos = None                # cached world position

    @property
    def pos(self):
        """World-space camera position on the orbit sphere (lazy recompute)."""
        if self._pos is None:
            yr, pr = math.radians(self.yaw), math.radians(self.pitch)
            offset = v3(math.sin(yr) * math.cos(pr),
                        math.sin(pr),
                        -math.cos(yr) * math.cos(pr)) * self.dist
            self._pos = self.target + offset
        return self._pos

    def orbit(self, dx, dy):
        """Blender-style MMB drag: dx spins the turntable, dy tilts."""
        self.yaw += dx * self.drag_speed
        self.pitch = max(self.pitch_min,
                         min(self.pitch_max, self.pitch - dy * self.drag_speed))
        self._pos = None

    def zoom(self, factor):
        self.dist = max(14.0, min(90.0, self.dist * factor))
        self._pos = None

    def tick(self, dt, dragging):
        """No auto-rotation: Blender-style, the view only moves when the user
        orbits with MMB.  Kept as a per-frame hook (currently a no-op)."""
        return

    def follow(self, target):
        """Keep the orbit target locked on the walking fly's body."""
        self.target = v3(target.x, target.y, target.z)
        self._pos = None


class FlyBody(object):
    """Procedural 3D wireframe fruit fly (Drosophila melanogaster).

    Body plan:  head (sphere wireframe), thorax (box), abdomen (tapered
    segments), 2 wings (fans that pivot for up/downstroke), 6 legs (3 segments
    each: coxa-femur, tibia, tarsus).  All joints are animated from motor
    neuron activations.
    """

    def __init__(self):
        self.time = 0.0
        # ------------------------------------------------------ head
        self.head_center = v3(0, 0.0, 6.0)
        self.head_r = 2.2
        # ------------------------------------------------------ thorax
        self.thorax = [v3(-2.2, -1.4, 3.2), v3(2.2, -1.4, 3.2),
                       v3(2.2, 1.4, 3.2), v3(-2.2, 1.4, 3.2),
                       v3(-2.2, -1.4, -2.2), v3(2.2, -1.4, -2.2),
                       v3(2.2, 1.4, -2.2), v3(-2.2, 1.4, -2.2)]
        # ------------------------------------------------------ abdomen (segments)
        self.abdomen_segs = []
        for i in range(5):
            w = 1.8 - i * 0.28
            h = 1.1 - i * 0.16
            z0 = -2.4 - i * 1.7
            self.abdomen_segs.append((v3(-w, -h, z0), v3(w, -h, z0),
                                      v3(w, h, z0), v3(-w, h, z0)))
        # ------------------------------------------------------ wings
        # hinge at thorax; wing is a fan of veins
        self.wing_L_hinge = v3(-1.9, 0.4, 0.4)
        self.wing_R_hinge = v3(1.9, 0.4, 0.4)
        self.wing_span = 11.0
        self.wing_chord = 5.0
        # ------------------------------------------------------ legs
        # attachment points on thorax: 3 per side (front / mid / hind)
        self.leg_roots_L = [v3(-1.8, -1.0, 2.6), v3(-2.0, -1.1, 0.5),
                            v3(-1.8, -1.0, -1.6)]
        self.leg_roots_R = [v3(1.8, -1.0, 2.6), v3(2.0, -1.1, 0.5),
                            v3(1.8, -1.0, -1.6)]
        # rest pose target angles
        self.leg_femur_len = 5.5
        self.leg_tibia_len = 4.5
        self.leg_tarsus_len = 3.0

        # ------------------ animation state driven by motor neurons ----------
        # EVERY joint below is a direct function of its motor-neuron pool:
        # nothing is clock-driven or scripted.  Quiet pools = still limbs.
        self.wing_phase_L = 0.0
        self.wing_phase_R = 0.0
        self.wing_drive_L = 0.0              # left VNC-T pool drive (0..1)
        self.wing_drive_R = 0.0              # right VNC-T pool drive (0..1)
        self.wing_amp_L = 0.0                # stroke amplitude, neural
        self.wing_amp_R = 0.0
        self.leg_lift_L = [0.0] * 3          # -1..1, from DN lift/place pools
        self.leg_lift_R = [0.0] * 3
        # which board squares command each leg (index = side*3 + k):
        # side = half of the board (files a-d / e-h), k = file % 3 pair
        self.leg_sq_idx = [self._leg_square_indices(s, k)
                           for s in (0, 1) for k in range(3)]
        self.abdomen_bend = 0.0
        self.head_yaw = 0.0
        # ------------------ locomotion: body in the world -------------------
        # The fly owns its own pose: DN walk/turn commands drive a tripod-gait
        # central pattern generator that moves this body around the arena.
        self.pos = v3(0, 0, 0)               # world position (arena floor y=0)
        self.heading = 0.0                   # body yaw in degrees
        self.walk_cmd = 0.0                  # forward command (0..1)
        self.turn_cmd = 0.0                  # yaw-rate command (-1..1)
        self.hold = False                    # posture hold while deliberating
        self.gait_phase = 0.0                # tripod central-pattern generator
        self.leg_swing = [0.0] * 6           # per-leg protraction (swing)
        self.leg_plant = [0.0] * 6           # per-leg stance extension
        self.eco = False                     # ECO: legacy scripted animation
        self.eco_t = 0.0                     # ECO: self-driving idle clock
        self.reward_buzz = 0.0               # phasic burst after DA reward
        self.leg_excite = 0.0
        self.eye_glow = 0.0
        self.flash_timer = 0.0               # green dopamine flash
        self.flash_text_timer = 0.0
        self.dopamine_anchor_y = 60.0
        self.dopamine_text = ""

    # ---- motor neuron mapping ----------------------------------------------
    @staticmethod
    def _leg_square_indices(side, k):
        """Board squares (0..63, a8..h1 row-major) whose DN drive commands the
        leg (side, k).  side 0 = left legs (files a-d), side 1 = right (e-h);
        k = file % 3 selects the front/mid/hind pair.  Each leg thus 'stands'
        on a distinct pool of squares and integrates their motor channels."""
        return [sq for sq in range(64)
                if (sq % 8 < 4) == (side == 0) and sq % 8 % 3 == k]

    def _decay_fx(self, dt):
        """Shared visual-fx timers (dopamine flash, eye glow) for both modes."""
        if self.flash_timer > 0:
            self.flash_timer = max(0.0, self.flash_timer - dt)
        if self.flash_text_timer > 0:
            self.flash_text_timer = max(0.0, self.flash_text_timer - dt)
            self.dopamine_anchor_y += dt * 26.0
        self.eye_glow = 0.25 + 0.75 * (self.flash_timer / 0.6
                                       if self.flash_timer > 0 else 0.0)

    def apply_motor(self, motor_acts, dt, hold=None):
        """Reflex motor layer: translate DN/VNC activations into limb pose.

        ARCHITECTURE (subsumption): this is the CHEAP innate layer - a handful
        of vector means and a fixed tripod CPG, no search, no learning, ~micro-
        seconds per frame.  All 'intellectual power' (iterative-deepening
        negamax + connectome propagation) is reserved for CHESS in
        FlyChessAgent.choose_move.  While the brain deliberates (hold=True)
        walk/turn commands decay to zero: the fly stands still and thinks,
        then commits to a motor program.

        Every moving joint reads its OWN motor-neuron pool; nothing is
        scripted or clock-driven, so quiet motoneurons mean a still fly:

          motor[0:64]    DN lift pools   -> 6 legs (half-board x file%3):
                            each leg's command = mean drive of its squares
          motor[64:128]  DN place pools  -> same 6 legs, opposing sign
          motor[128:144] VNC-T left      -> left wing stroke rate + amplitude
          motor[144:160] VNC-T right     -> right wing stroke rate + amplitude
          motor[160:176] VNC-A left      } two turning pools: their asymmetry
          motor[176:192] VNC-A right     } gives abdomen bend + head yaw + turn
          place-pool mean                -> forward walk command (tripod gait)
        """
        if self.eco:
            # ECO MODE: legacy hard-coded animation.  Fixed clocks drive the
            # limbs - no motor-neuron readout, no gait CPG, no walking, no
            # proprioceptive sensing - so the connectome's compute budget is
            # spent on chess deliberation only.
            self.eco_t += dt
            t = self.eco_t
            self.wing_phase_L += dt * 40.0
            self.wing_phase_R += dt * 40.0
            self.wing_amp_L = self.wing_amp_R = 1.0
            for i in range(3):
                self.leg_lift_L[i] = 0.25 * math.sin(t * 6.0 + i)
                self.leg_lift_R[i] = 0.25 * math.sin(t * 6.0 + i + 0.5)
            for i in range(6):
                self.leg_swing[i] = 0.0
                self.leg_plant[i] = 0.0
            self.walk_cmd = self.turn_cmd = 0.0
            self.gait_phase = 0.0
            self.abdomen_bend = 8.0 * math.sin(t * 1.5)
            self.head_yaw = 10.0 * math.sin(t * 0.9)
            self.leg_excite = 0.0
            self._decay_fx(dt)
            return

        m = np.asarray(motor_acts, dtype=float)
        lift, place = m[:64], m[64:128]

        # --- legs: all six integrate their own square-pool commands ---------
        for s in (0, 1):
            for k in range(3):
                idxs = self.leg_sq_idx[s * 3 + k]
                lift_cmd = float(np.mean(lift[idxs])) if idxs else 0.0
                place_cmd = float(np.mean(place[idxs])) if idxs else 0.0
                target = float(np.clip(1.4 * lift_cmd - 1.4 * place_cmd,
                                       -1.0, 1.0))
                arr = self.leg_lift_R if s else self.leg_lift_L
                arr[k] += (target - arr[k]) * min(1.0, dt * 9.0)

        # --- wings: each wing reads its own VNC-T half ----------------------
        wl = float(np.mean(np.abs(m[128:144]))) if m.size >= 160 else 0.0
        wr = float(np.mean(np.abs(m[144:160]))) if m.size >= 160 else 0.0
        self.wing_drive_L += (float(np.clip(wl * 2.0, 0, 1))
                              - self.wing_drive_L) * min(1.0, dt * 8.0)
        self.wing_drive_R += (float(np.clip(wr * 2.0, 0, 1))
                              - self.wing_drive_R) * min(1.0, dt * 8.0)
        # BIOLOGY: a phasic dopamine burst (reward) adds wing power, like the
        # courtship-song buzz - the one neuromodulatory input on top of the
        # sensory->motor drive.
        self.reward_buzz = max(0.0, self.reward_buzz - dt * 0.8)
        self.wing_amp_L = min(1.25, 0.88 * self.wing_drive_L
                              + 0.5 * self.reward_buzz)
        self.wing_amp_R = min(1.25, 0.88 * self.wing_drive_R
                              + 0.5 * self.reward_buzz)
        self.wing_phase_L += dt * (54.0 * self.wing_drive_L
                                   + 30.0 * self.reward_buzz)
        self.wing_phase_R += dt * (54.0 * self.wing_drive_R
                                   + 30.0 * self.reward_buzz)

        # --- abdomen + head: VNC-A left/right asymmetry = turn torque -------
        al = float(np.mean(m[160:176])) if m.size >= 192 else 0.0
        ar = float(np.mean(m[176:192])) if m.size >= 192 else 0.0
        asym, total = al - ar, 0.5 * (al + ar)
        bend_target = (float(np.clip(total * 30.0, -25, 25))
                       + float(np.clip(asym * 20.0, -15, 15)))
        self.abdomen_bend += (bend_target - self.abdomen_bend) * min(1.0, dt * 6.0)
        head_target = float(np.clip(asym * 40.0, -30, 30))
        self.head_yaw += (head_target - self.head_yaw) * min(1.0, dt * 5.0)

        # --- locomotion: innate VNC reflex, NOT part of the chess brain -----
        # BIOLOGY: fruit flies walk with a tripod gait; descending-neuron
        # population drive sets forward speed and left/right steering.
        # COMPUTATION: place-pool mean = "go" command, VNC-A asymmetry =
        # yaw-rate command.  Deliberation (hold) freezes locomotion so the
        # expensive search never competes with body control.
        if hold is None:
            hold = self.hold
        k_cmd = min(1.0, dt * (10.0 if hold else 5.0))
        fwd = 0.0 if hold else float(np.mean(m[64:128]))
        turn_asym = 0.0 if hold else asym
        walk_target = 0.0 if hold else float(np.clip(2.0 * fwd, 0.0, 1.0))
        turn_target = 0.0 if hold else float(np.clip(turn_asym * 3.0, -1.0, 1.0))
        self.walk_cmd += (walk_target - self.walk_cmd) * k_cmd
        self.turn_cmd += (turn_target - self.turn_cmd) * min(1.0, dt * 6.0)

        # --- body physics: tripod-gait CPG + heading + position ------------
        self.gait_phase += dt * 9.0 * self.walk_cmd
        self.heading += self.turn_cmd * 160.0 * self.walk_cmd * dt
        speed = 9.0 * self.walk_cmd
        hr = math.radians(self.heading)
        self.pos = self.pos + v3(math.sin(hr) * speed * dt, 0.0,
                                 math.cos(hr) * speed * dt)
        for i in range(6):
            # tripods: {L-front, L-hind, R-mid} vs {L-mid, R-front, R-hind}
            gp = (self.gait_phase + (0.5 if i % 2 else 0.0)) % 1.0
            if gp < 0.35:
                swing = math.sin(gp / 0.35 * math.pi)      # protraction
                plant = 0.0
            else:
                swing = 0.0
                plant = 1.0                                # stance
            self.leg_swing[i] = swing * (0.35 + 0.65 * self.walk_cmd)
            self.leg_plant[i] = plant

        # peak DN drive (used only for the claw highlight colour)
        self.leg_excite = min(1.0, float(np.max(np.abs(m[:128])))) if m.size else 0.0

        # --- dopamine flash decay -----------------------------------------
        self._decay_fx(dt)

    def trigger_reward(self, text="+DOPAMINE"):
        self.flash_timer = 0.6
        self.flash_text_timer = 1.4
        self.dopamine_text = text
        self.dopamine_anchor_y = 60.0
        self.reward_buzz = 1.0        # dopamine burst -> wing buzz accelerates

    def trigger_penalty(self):
        self.flash_timer = 0.0
        self.flash_text_timer = 1.2
        self.dopamine_text = "-DA (DEPRESSION)"
        self.dopamine_anchor_y = 60.0
        self.reward_buzz = 0.0                # wings droop: no burst

    # ------------------------------- geometry --------------------------------
    def _to_world(self, p_local):
        """Local body coordinates -> world: body yaw (heading) + position."""
        return rot_around(p_local, (0, 1, 0), self.heading) + self.pos

    def _head_points(self, root):
        pts = []
        lat, lon = 8, 10
        for i in range(lat + 1):
            th = math.pi * i / lat
            for j in range(lon):
                ph = 2 * math.pi * j / lon
                x = self.head_r * math.sin(th) * math.cos(ph)
                y = self.head_r * math.cos(th)
                z = self.head_r * math.sin(th) * math.sin(ph)
                pts.append(v3(x, y, z) + root)
        # ring index helper
        rings = []
        for i in range(lat + 1):
            rings.append(list(range(i * lon, (i + 1) * lon)))
        return pts, rings

    def draw(self, surf, cam, sensory_acts=None):
        # collect edges as (depth, p1, p2, colour)
        edges = []
        self._draw_head(surf, edges, cam)
        self._draw_thorax(edges, cam)
        self._draw_abdomen(edges, cam)
        self._draw_wings(edges, cam)
        self._draw_legs(edges, cam)
        # painter's algorithm: draw far edges first
        edges.sort(key=lambda e: e[0], reverse=True)
        for depth, p1, p2, color in edges:
            pygame.draw.line(surf, color, p1, p2, 1)
        # sensory columns as a floating 8x8 LED grid above the fly
        if sensory_acts is not None:
            self._draw_sensory_grid(surf, cam, sensory_acts)
        self._draw_dopamine_anchor(surf)

    # ----- body parts --------------------------------------------------------
    def _seg(self, edges, cam, a, b, color, lift=0.0):
        p1, d1 = project(a, cam)
        p2, d2 = project(b, cam)
        edges.append(((d1 + d2) * 0.5, p1, p2, color))

    def _draw_head(self, surf, edges, cam):
        root = self._to_world(self.head_center)
        # neck yaw (head_yaw) then the whole-body world transform
        pts, rings = self._head_points(v3(0, 0, 0))
        yawed = [self._to_world(rot_around(p, (0, 1, 0), self.head_yaw)
                                + self.head_center) for p in pts]
        for ring in rings:
            for i in range(len(ring)):
                a = yawed[ring[i]]
                b = yawed[ring[(i + 1) % len(ring)]]
                self._seg(edges, cam, a, b, CYAN if self.eye_glow > 0.5 else GREY)
        # two big compound eyes as glowing circles (billboards)
        for sx in (-1, 1):
            c = self._to_world(self.head_center + v3(sx * 1.3, 0.4, 1.6))
            sp, _ = project(c, cam)
            col = (int(255 * self.eye_glow), 80, 80)
            pygame.draw.circle(surf, col, sp, 6)
        # antennae
        for sx in (-1, 1):
            base = root + v3(sx * 0.7, 1.2, 1.8)
            tip = base + v3(sx * 1.6, 1.8, 1.4) + v3(0, math.sin(self.time * 9 + sx) * 0.5, 0)
            self._seg(edges, cam, base, tip, GREY)
        # proboscis
        pb = root + v3(0, -0.8, 1.9)
        pt = pb + v3(0, -1.2, 1.0)
        self._seg(edges, cam, pb, pt, GREY)

    def _draw_thorax(self, edges, cam):
        th = self.thorax
        # rotate the whole body block by abdomen bend around Y for turning
        pts = [self._to_world(rot_around(p - v3(0, 0, 0.5),
                                         (0, 1, 0), self.abdomen_bend * 0.3)
                              + v3(0, 0, 0.5)) for p in th]
        idx_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                     (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in idx_pairs:
            self._seg(edges, cam, pts[i], pts[j], AMBER)
        self.thorax_world = pts

    def _draw_abdomen(self, edges, cam):
        prev = None
        for si, (a, b, c, d) in enumerate(self.abdomen_segs):
            swing = self.abdomen_bend * 0.12 * (si + 1)
            q = [self._to_world(rot_around(p, (0, 1, 0), swing))
                 for p in (a, b, c, d)]
            ring_idx = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for i, j in ring_idx:
                self._seg(edges, cam, q[i], q[j], GREEN if si % 2 == 0 else (40, 160, 80))
            if prev is not None:
                for k in range(4):
                    self._seg(edges, cam, prev[k], q[k], (40, 160, 80))
            prev = q

    def _draw_wings(self, edges, cam):
        # stroke speed and amplitude come straight from each wing's own
        # VNC-T motoneuron pool (see apply_motor); dopamine adds a burst
        for side in (-1, 1):
            hinge = self.wing_L_hinge if side < 0 else self.wing_R_hinge
            phase = self.wing_phase_L if side < 0 else self.wing_phase_R
            amp = self.wing_amp_L if side < 0 else self.wing_amp_R
            # up/downstroke scaled by neural amplitude + fine buzz shimmer
            stroke = math.sin(phase) * 55.0 * amp
            buzz = math.sin(phase * 7.0) * 6.0 * amp
            ang = stroke + buzz
            # wing membrane fan: 5 veins
            for vi in range(5):
                t = vi / 4.0
                chord_off = (t - 0.5) * self.wing_chord
                span = self.wing_span * (1.0 - 0.25 * abs(t - 0.5) * 2)
                tip = v3(side * span, chord_off * 0.6, 0.0)
                tip = rot_around(tip, (0, 0, 1), ang * side)
                tip = rot_around(tip, (1, 0, 0), 18.0)
                world = self._to_world(hinge + tip)
                self._seg(edges, cam, self._to_world(hinge), world,
                          VIOLET if vi else WHITE)
            # wing outline (leading + trailing edge)
            lead = hinge + rot_around(v3(side * self.wing_span, -self.wing_chord * 0.45, 0), (0, 0, 1), ang * side)
            lead = rot_around(lead - hinge, (1, 0, 0), 18.0) + hinge
            trail = hinge + rot_around(v3(side * self.wing_span * 0.75, self.wing_chord * 0.5, 0), (0, 0, 1), ang * side)
            trail = rot_around(trail - hinge, (1, 0, 0), 18.0) + hinge
            wl = self._to_world(hinge)
            self._seg(edges, cam, wl, self._to_world(lead), VIOLET)
            self._seg(edges, cam, self._to_world(lead), self._to_world(trail),
                      VIOLET)
            self._seg(edges, cam, self._to_world(trail), wl, VIOLET)

    def _draw_legs(self, edges, cam):
        # pose = this leg's own motor commands (lift/place) + gait swing/plant
        for side in (-1, 1):
            roots = self.leg_roots_L if side < 0 else self.leg_roots_R
            lifts = self.leg_lift_L if side < 0 else self.leg_lift_R
            for li, root in enumerate(roots):
                lift = lifts[li]
                li6 = (0 if side < 0 else 3) + li
                swing = self.leg_swing[li6]        # protraction (swing phase)
                plant = self.leg_plant[li6]        # stance loading
                femur = v3(side * 3.4, -2.2 + lift * 2.6,
                           0.6 * math.sin(li * 1.7) + swing * 2.2)
                f_end = root + femur
                # tibia
                tib = v3(side * 1.4, -2.6 - lift * 1.4 + plant * 0.9, -1.1)
                t_end = f_end + tib
                # tarsus
                tar = v3(side * 0.7, -1.5 - lift * 0.5, -1.0)
                a_end = t_end + tar
                col = AMBER if abs(lift) > 0.45 else GREY
                wroot = self._to_world(root)
                self._seg(edges, cam, wroot, self._to_world(f_end), col)
                self._seg(edges, cam, self._to_world(f_end),
                          self._to_world(t_end), col)
                self._seg(edges, cam, self._to_world(t_end),
                          self._to_world(a_end), col)
                # little claw
                self._seg(edges, cam, self._to_world(a_end),
                          self._to_world(a_end + v3(side * 0.3, -0.6, -0.3)),
                          col)

    # ----- overlays -----------------------------------------------------------
    def _draw_sensory_grid(self, surf, cam, sensory):
        """8x8 'retina' showing current ME(R) column activations."""
        ox, oy = 24, 40
        cell = 13
        for idx, act in enumerate(sensory):
            r, c = divmod(idx, 8)
            a = float(np.clip(abs(act), 0, 1))
            if act < 0:
                col = (int(60 + 120 * a), int(60 - 30 * a), int(200))
            else:
                col = (int(60 + 195 * a), int(60 + 170 * a), 60)
            pygame.draw.rect(surf, col,
                             (ox + c * cell, oy + r * cell, cell - 2, cell - 2),
                             border_radius=3)
        f = pygame.font.SysFont("consolas", 14)
        surf.blit(f.render("ME(R) sensory columns (8x8 retinotopy)", True, GREY),
                  (ox, oy + 8 * cell + 6))

    def _draw_dopamine_anchor(self, surf):
        if self.flash_text_timer <= 0:
            return
        f = pygame.font.SysFont("consolas", 30, bold=True)
        col = GREEN if self.flash_timer > 0 else RED
        txt = f.render(self.dopamine_text, True, col)
        surf.blit(txt, (VIEW3D_CX - txt.get_width() // 2,
                        int(self.dopamine_anchor_y)))


class BoardPanel(object):
    """Live 2D chessboard on the right half of the window.

    Shows the current game state with unicode pieces, highlights the last
    move and the connectome's from/to squares, and converts mouse clicks to
    board squares so a human can play against the fly.
    """

    LIGHT = (222, 210, 178)
    DARK = (140, 115, 85)
    HL_LAST = (120, 200, 120)
    HL_SELECT = (240, 220, 90)
    HL_FLY = (150, 110, 255)

    def __init__(self):
        self.x0 = BOARD_X0
        self.y0 = BOARD_Y0
        self.selected = None            # chess square or None
        self.legal_targets = set()
        self.promotion_pending = None  # (from_square, to_square)
        self.pending_move = None
        self.history = []              # SAN strings, oldest first
        self.captured_white = []       # white pieces captured by Black
        self.captured_black = []       # black pieces captured by White
        self.drag_from = None
        self.drag_pos = None
        self.drag_piece = None
        self.drag_moved = False
        self.arrow_from = None
        self.arrow_pos = None
        self.arrow_dragging = False
        self.analysis_arrows = []
        self.piece_font = None
        self._piece_cache = {}
        self.ui_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 17, bold=True)
        self.clock_font = pygame.font.SysFont("consolas", 22, bold=True)
        self.history_font = pygame.font.SysFont("consolas", 13)
        self.banner_font = pygame.font.SysFont("consolas", 13, bold=True)
        self.capture_font = pygame.font.SysFont("consolas", 13)
        self.prompt_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.status_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.animation_segments = []
        self.animation_started = 0.0
        self.animation_duration = 0.0
        self.animation_source = None
        self.animation_capture_square = None

    def set_full_screen(self, enabled):
        self.x0 = 560 if enabled else BOARD_X0
        self.y0 = 110

    def reset(self):
        self.selected = None
        self.legal_targets = set()
        self.promotion_pending = None
        self.pending_move = None
        self.history = []
        self.captured_white = []
        self.captured_black = []
        self.drag_from = None
        self.drag_pos = None
        self.drag_piece = None
        self.drag_moved = False
        self.arrow_from = None
        self.arrow_pos = None
        self.arrow_dragging = False
        self.analysis_arrows = []
        self.animation_segments = []
        self.animation_started = 0.0
        self.animation_duration = 0.0
        self.animation_source = None
        self.animation_capture_square = None

    def begin_move_animation(self, before, after, move, duration=0.24):
        """Animate a just-played move over the post-move position."""
        piece_before = before.piece_at(move.from_square)
        piece_after = after.piece_at(move.to_square)
        if piece_before is None or piece_after is None:
            self.animation_segments = []
            return
        segments = [{"from": move.from_square, "to": move.to_square,
                     "piece": piece_after}]
        if before.is_castling(move):
            rank = chess.square_rank(move.from_square)
            if before.is_kingside_castling(move):
                rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
            else:
                rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
            rook = before.piece_at(rook_from)
            if rook is not None:
                segments.append({"from": rook_from, "to": rook_to,
                                 "piece": rook})
        self.animation_segments = segments
        self.animation_source = move.from_square
        self.animation_capture_square = (chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square))
            if before.is_en_passant(move) else move.to_square)
        self.animation_started = time.perf_counter()
        self.animation_duration = max(0.08, float(duration))

    def animation_active(self):
        if not self.animation_segments:
            return False
        if time.perf_counter() - self.animation_started >= self.animation_duration:
            self.animation_segments = []
            self.animation_source = None
            self.animation_capture_square = None
            return False
        return True

    def record_move(self, board, move):
        """Record SAN, captures, and promotion history before pushing `move`."""
        self.history.append(board.san(move))
        victim = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board.turn)
        if victim:
            (self.captured_white if victim.color == chess.WHITE
             else self.captured_black).append(victim.symbol().lower())

    def handle_key(self, key):
        """Choose a pending promotion with Q/R/B/N; return the move if ready."""
        if self.promotion_pending is None:
            return None
        promo = {pygame.K_q: chess.QUEEN, pygame.K_r: chess.ROOK,
                 pygame.K_b: chess.BISHOP, pygame.K_n: chess.KNIGHT}.get(key)
        if promo is None:
            return None
        a, b = self.promotion_pending
        self.promotion_pending = None
        self.pending_move = chess.Move(a, b, promotion=promo)
        self.selected, self.legal_targets = None, set()
        return self.pending_move

    def square_rect(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x = self.x0 + f * SQ_PX
        y = self.y0 + (7 - r) * SQ_PX
        return (x, y, SQ_PX, SQ_PX)

    def square_at(self, mx, my):
        """Screen coords -> chess square (or None if outside the board)."""
        if not (self.x0 <= mx < self.x0 + 8 * SQ_PX):
            return None
        if not (self.y0 <= my < self.y0 + 8 * SQ_PX):
            return None
        f = (mx - self.x0) // SQ_PX
        r = 7 - (my - self.y0) // SQ_PX
        return chess.square(int(f), int(r))

    def begin_drag(self, board, pos, button=1):
        """Pick up a piece, or begin a right-button analysis arrow."""
        sq = self.square_at(*pos)
        if button == 3:
            if sq is not None:
                if self.arrow_from is not None and sq != self.arrow_from:
                    self.arrow_pos = tuple(pos)
                else:
                    self.arrow_from = sq
                    self.arrow_pos = tuple(pos)
                    self.arrow_dragging = True
            return sq is not None
        if sq is None or not (board.piece_at(sq) and
                              board.piece_at(sq).color == chess.WHITE):
            return False
        self.selected = sq
        self.legal_targets = {m.to_square for m in board.legal_moves
                              if m.from_square == sq}
        self.drag_from = sq
        self.drag_pos = tuple(pos)
        self.drag_piece = board.piece_at(sq)
        self.drag_moved = False
        return True

    def drag_to(self, pos):
        if self.drag_from is not None:
            self.drag_pos = tuple(pos)
            self.drag_moved = True
        if self.arrow_from is not None:
            self.arrow_pos = tuple(pos)

    def end_arrow(self, pos):
        if self.arrow_from is None:
            return
        target = self.square_at(*pos)
        source = self.arrow_from
        if target is None or target == source:
            # A right-click on the source alone keeps the source armed so the
            # next right-click can complete a two-click analysis arrow.
            return target
        self.arrow_from = None
        self.arrow_pos = None
        self.arrow_dragging = False
        pair = (source, target)
        if pair in self.analysis_arrows:
            self.analysis_arrows.remove(pair)
        else:
            self.analysis_arrows.append(pair)
        return target

    def end_drag(self, board, pos):
        """Drop a held piece; legal drops become a Move, invalid drops snap back."""
        if self.drag_from is None:
            return self.handle_click(board, pos)
        source = self.drag_from
        self.drag_from = self.drag_pos = self.drag_piece = None
        moved = self.drag_moved
        self.drag_moved = False
        target = self.square_at(*pos)
        if target is None or target not in self.legal_targets:
            if not moved:
                return self.handle_click(board, pos)
            return None
        choices = [m for m in board.legal_moves
                   if m.from_square == source and m.to_square == target]
        if len(choices) > 1:
            self.promotion_pending = (source, target)
            return None
        self.selected, self.legal_targets = None, set()
        return choices[0] if choices else None

    def handle_click(self, board, pos):
        """Click-to-select / click-to-move.  Returns a Move or None.

        White (the human) clicks one of their pieces, then a destination.
        """
        sq = self.square_at(*pos)
        if sq is None:
            self.selected, self.legal_targets = None, set()
            return None
        piece = board.piece_at(sq)
        if self.selected is not None and sq in self.legal_targets:
            choices = [mv for mv in board.legal_moves
                       if mv.from_square == self.selected and mv.to_square == sq]
            if len(choices) > 1:
                self.promotion_pending = (self.selected, sq)
                return None
            if choices:
                self.selected, self.legal_targets = None, set()
                return choices[0]
        if piece and piece.color == chess.WHITE:
            self.selected = sq
            self.legal_targets = {m.to_square for m in board.legal_moves
                                  if m.from_square == sq}
        else:
            self.selected, self.legal_targets = None, set()
        return None

    def _piece_surfaces(self, piece):
        if self.piece_font is None:
            self.piece_font = pygame.font.SysFont(
                "segoeuisymbol,dejavusans,arial", SQ_PX - 14)
        glyph = piece.unicode_symbol().lower()
        cache_key = (glyph, piece.color)
        cached = self._piece_cache.get(cache_key)
        if cached is None:
            if piece.color == chess.WHITE:
                glyph_col, shadow_col = (250, 250, 250), (30, 30, 30)
            else:
                glyph_col, shadow_col = (28, 28, 34), (215, 215, 255)
            cached = (self.piece_font.render(glyph, True, shadow_col),
                      self.piece_font.render(glyph, True, glyph_col))
            self._piece_cache[cache_key] = cached
        return cached

    def _arrow(self, surf, move, color=(80, 205, 120), width=8):
        """Draw a clean, manual analysis arrow only."""
        if move is None:
            return
        fx, fy, fw, fh = self.square_rect(move.from_square)
        tx, ty, tw, th = self.square_rect(move.to_square)
        layer = pygame.Surface((8 * SQ_PX, 8 * SQ_PX), pygame.SRCALPHA)
        start = (fx + fw // 2 - self.x0, fy + fh // 2 - self.y0)
        end = (tx + tw // 2 - self.x0, ty + th // 2 - self.y0)
        self._draw_arrow_points(layer, start, end, color, width)
        surf.blit(layer, (self.x0, self.y0))

    @staticmethod
    def _draw_arrow_points(surf, start, end, color=(80, 205, 120), width=8):
        """Draw one polished arrow into a shared transparent overlay."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 5.0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        # Stop the shaft short of the head so it does not poke through it.
        head_len = min(24.0, max(14.0, length * 0.22))
        shaft_end = (int(end[0] - ux * head_len),
                     int(end[1] - uy * head_len))
        rgba = (*color[:3], 205)
        pygame.draw.line(surf, rgba, start, shaft_end, width)
        pygame.draw.circle(surf, rgba, start, max(4, width // 2))
        head_w = max(10.0, width * 1.45)
        left = (int(end[0] - ux * head_len + px * head_w),
                int(end[1] - uy * head_len + py * head_w))
        right = (int(end[0] - ux * head_len - px * head_w),
                 int(end[1] - uy * head_len - py * head_w))
        pygame.draw.polygon(surf, rgba, [end, left, right])

    def draw(self, surf, board, last_move=None, fly_move=None,
             status=None, game_over=None, clocks=None, review=False,
             review_analysis=None, review_san_moves=None, review_index=0):
        now = time.perf_counter()
        animating = bool(self.animation_segments and
                         now - self.animation_started < self.animation_duration)
        if self.animation_segments and not animating:
            self.animation_segments = []
            self.animation_source = None
            self.animation_capture_square = None
        animated_destinations = ({segment["to"]
                                  for segment in self.animation_segments}
                                 if animating else set())
        animated_sources = ({segment["from"]
                             for segment in self.animation_segments}
                            if animating else set())
        # board frame + squares
        pygame.draw.rect(surf, GREY,
                         (self.x0 - 2, self.y0 - 2,
                          8 * SQ_PX + 4, 8 * SQ_PX + 4), 2)
        for r in range(8):
            for f in range(8):
                sq = chess.square(f, r)
                col = self.LIGHT if (f + r) % 2 else self.DARK
                x, y, w, h = self.square_rect(sq)
                pygame.draw.rect(surf, col, (x, y, w, h))
                if last_move and sq in (last_move.from_square,
                                        last_move.to_square):
                    pygame.draw.rect(surf, self.HL_LAST, (x, y, w, h), 3)
                if fly_move and sq in (fly_move.from_square,
                                       fly_move.to_square):
                    pygame.draw.rect(surf, self.HL_FLY, (x, y, w, h), 3)
                if sq in self.legal_targets:
                    pygame.draw.circle(surf, self.HL_SELECT,
                                       (x + w // 2, y + h // 2),
                                       max(4, SQ_PX // 8))
                if sq == self.selected:
                    pygame.draw.rect(surf, self.HL_SELECT, (x, y, w, h), 3)
                piece = board.piece_at(sq)
                hidden_move_source = (animating and
                                      (sq in animated_sources
                                       or sq == self.animation_source))
                hidden_capture = (animating and
                                  sq == self.animation_capture_square)
                if piece and sq not in animated_destinations and \
                        not hidden_move_source and not hidden_capture:
                    # All pieces use the FILLED unicode glyphs (♟♞♝♜♛♚);
                    # colour + outline shadow distinguishes the sides, so both
                    # white and black pieces always render with a real sprite.
                    filled = piece.unicode_symbol().lower()
                    sh, ts = self._piece_surfaces(piece)
                    px = x + (SQ_PX - ts.get_width()) // 2
                    py = y + (SQ_PX - ts.get_height()) // 2
                    surf.blit(sh, (px + 2, py + 2))      # outline shadow
                    surf.blit(ts, (px, py))
        # Only manually drawn analysis arrows appear here. Played chess moves
        # intentionally use square highlights, never automatic arrows. Draw
        # all arrows into one board-sized layer to avoid a full-window surface
        # allocation per arrow on every animation frame.
        arrow_specs = []
        for arrow_from, arrow_to in self.analysis_arrows:
            fx, fy, fw, fh = self.square_rect(arrow_from)
            tx, ty, tw, th = self.square_rect(arrow_to)
            arrow_specs.append(((fx + fw // 2 - self.x0,
                                 fy + fh // 2 - self.y0),
                                (tx + tw // 2 - self.x0,
                                 ty + th // 2 - self.y0)))
        if self.arrow_from is not None and self.arrow_pos is not None:
            fx, fy, fw, fh = self.square_rect(self.arrow_from)
            start = (fx + fw // 2 - self.x0, fy + fh // 2 - self.y0)
            target = self.square_at(*self.arrow_pos)
            if target is not None and target != self.arrow_from:
                tx, ty, tw, th = self.square_rect(target)
                end = (tx + tw // 2 - self.x0, ty + th // 2 - self.y0)
            else:
                end = (self.arrow_pos[0] - self.x0,
                       self.arrow_pos[1] - self.y0)
            arrow_specs.append((start, end))
        if arrow_specs:
            arrow_layer = pygame.Surface((8 * SQ_PX, 8 * SQ_PX),
                                         pygame.SRCALPHA)
            for start, end in arrow_specs:
                self._draw_arrow_points(arrow_layer, start, end)
            surf.blit(arrow_layer, (self.x0, self.y0))

        if animating:
            progress = min(1.0, max(0.0,
                (now - self.animation_started) / self.animation_duration))
            eased = progress * progress * (3.0 - 2.0 * progress)
            for segment in self.animation_segments:
                sx, sy, sw, sh = self.square_rect(segment["from"])
                tx, ty, tw, th = self.square_rect(segment["to"])
                start = (sx + sw // 2, sy + sh // 2)
                end = (tx + tw // 2, ty + th // 2)
                center = (round(start[0] + (end[0] - start[0]) * eased),
                          round(start[1] + (end[1] - start[1]) * eased))
                shad, sprite = self._piece_surfaces(segment["piece"])
                sprite_rect = sprite.get_rect(center=center)
                surf.blit(shad, sprite_rect.move(2, 2))
                surf.blit(sprite, sprite_rect)

        # Drag ghost: the original square is dimmed and a translucent piece
        # follows the cursor until the drop is accepted.
        if self.drag_piece is not None and self.drag_pos is not None:
            ghost_font = self.piece_font or pygame.font.SysFont(
                "segoeuisymbol,dejavusans,arial", SQ_PX - 10)
            glyph = self.drag_piece.unicode_symbol().lower()
            self.piece_font = ghost_font
            ghost = ghost_font.render(glyph, True, (255, 255, 255))
            ghost.set_alpha(220)
            rect = ghost.get_rect(center=self.drag_pos)
            surf.blit(ghost, rect)
        if self.promotion_pending:
            prompt = self.prompt_font
            surf.blit(prompt.render("PROMOTE: Q  R  B  N", True, AMBER),
                      (self.x0 + 220, self.y0 + 8 * SQ_PX + 26))

        # pane title + coordinates + caption
        fnt = self.ui_font
        title = self.title_font
        surf.blit(title.render("REPLAY BOARD" if review else "CHESSBOARD PANE",
                               True, CYAN), (self.x0, self.y0 - 46))
        banner = self.banner_font
        banner_text = ("reviewing saved PGN" if review else
                       "DRAG pieces  |  right-drag marks plans  |  click also works")
        surf.blit(banner.render(banner_text, True, AMBER),
                  (self.x0 + 250, self.y0 - 46))
        caption_text = ("Use the replay controls to browse this game" if review else
                        "PLAY THE FLY - click a white piece, then a square")
        surf.blit(fnt.render(caption_text, True, WHITE),
                  (self.x0, self.y0 - 22))
        for i in range(8):
            surf.blit(fnt.render(FILES[i], True, GREY),
                      (self.x0 + i * SQ_PX + SQ_PX // 2 - 4,
                       self.y0 + 8 * SQ_PX + 6))
            surf.blit(fnt.render(str(i + 1), True, GREY),
                      (self.x0 - 14, self.y0 + (7 - i) * SQ_PX + SQ_PX // 2 - 7))
        # Clocks, captures, and a compact SAN history make this a usable
        # training board rather than only a visualization.
        if clocks:
            clock_font = self.clock_font
            def fmt(sec):
                sec = max(0, int(sec))
                return "%02d:%02d" % (sec // 60, sec % 60)
            surf.blit(clock_font.render("WHITE " + fmt(clocks.get("white", 0)),
                                        True, WHITE), (self.x0, 18))
            surf.blit(clock_font.render("BLACK " + fmt(clocks.get("black", 0)),
                                        True, (40, 40, 45)),
                      (self.x0 + 170, 18))
        cap_font = self.capture_font
        surf.blit(cap_font.render(
            "captured by White: " + (" ".join(self.captured_black) or "-"),
            True, WHITE), (self.x0 + 360, 22))
        surf.blit(cap_font.render(
            "captured by Black: " + (" ".join(self.captured_white) or "-"),
            True, (40, 40, 45)), (self.x0 + 360, 42))
        history_y = self.y0 + 8 * SQ_PX + 58
        hist_font = self.history_font
        if review:
            review_sans = (review_san_moves if review_san_moves is not None
                           else self.history)
            total_plies = len(review_sans)
            current_ply = review_index - 1
            first_ply = (max(0, current_ply) // 16) * 16
            first_ply = min(first_ply, max(0, total_plies - 1) // 16 * 16)
            last_ply = min(first_ply + 16, total_plies)
            if total_plies:
                section_label = "MOVE GRADES · PLIES %d-%d OF %d" % (
                    first_ply + 1, last_ply, total_plies)
            else:
                section_label = "MOVE GRADES · NO MOVES"
            surf.blit(hist_font.render(section_label, True, (139, 159, 181)),
                      (self.x0, history_y - 18))
            row_step = hist_font.get_linesize() + 3
            for row_index, row_start in enumerate(range(
                    first_ply, last_ply, 4)):
                x = self.x0
                y = history_y + row_index * row_step
                for ply in range(row_start, min(row_start + 4, last_ply)):
                    prefix = ("%d. " % (ply // 2 + 1)
                              if ply % 2 == 0 else "")
                    move_color = ((100, 218, 231) if ply == current_ply
                                  else (229, 237, 246))
                    move_text = hist_font.render(
                        prefix + review_sans[ply], True, move_color)
                    surf.blit(move_text, (x, y))
                    x += move_text.get_width() + 5
                    analysis = (review_analysis.get(ply)
                                if review_analysis else None)
                    quality = analysis.get("quality") if analysis else None
                    if quality:
                        color = analysis.get(
                            "quality_color",
                            STOCKFISH_MOVE_QUALITY_COLORS.get(
                                quality, STOCKFISH_MOVE_QUALITY_COLORS["Good"]))
                        badge_text = hist_font.render(
                            quality.upper(), True, (18, 27, 35))
                        badge = pygame.Rect(x, y - 1,
                                            badge_text.get_width() + 8,
                                            hist_font.get_linesize() + 1)
                        pygame.draw.rect(surf, color, badge, border_radius=4)
                        surf.blit(badge_text, (badge.x + 4, y))
                        x += badge.width + 8
                    else:
                        x += 4
        else:
            recent = self.history[-16:]
            pairs = []
            for i in range(0, len(recent), 2):
                pair = "%d. %s" % (len(self.history) // 2 + i // 2 + 1,
                                    recent[i])
                if i + 1 < len(recent):
                    pair += " " + recent[i + 1]
                pairs.append(pair)
            surf.blit(hist_font.render("HISTORY  " + (" | ".join(pairs) or "-"),
                                       True, WHITE), (self.x0, history_y))

        # status / game-over ribbon
        if status:
            big = self.status_font
            txt_col = GREEN if game_over in (None, "human") else RED
            if game_over == "fly":
                txt_col = VIOLET
            ts = big.render(status, True, txt_col)
            surf.blit(ts, (self.x0 + 4 * SQ_PX - ts.get_width() // 2,
                           self.y0 + 8 * SQ_PX + 26))


def draw_hud(surf, info):
    f = pygame.font.SysFont("consolas", 15)
    lines = [
        "FRUIT FLY CONNECTOME CHESS  |  %s%s"
        % (info.get("mode", ""), "  [ECO]" if info.get("eco") else ""),
        "neuPrint: %s   dataset: %s" % (info.get("neuprint", "?"),
                                        info.get("dataset", "?")),
        "game %d/%d   ply %d   move: %s" % (info.get("game", 0),
                                            info.get("games", 0),
                                            info.get("ply", 0),
                                            info.get("last_move", "-")),
        "fly score: %.1f   DA spikes: %d (reward) / %d (penalty)" % (
            info.get("fly_score", 0.0), info.get("da_reward", 0),
            info.get("da_penalty", 0)),
        "mean |W|: %.3f   dopamine last: %+.3f" % (info.get("mean_w", 0.0),
                                                   info.get("da_last", 0.0)),
        "walk: %.2f   heading: %.0f   pos: (%.1f, %.1f)" % (
            info.get("walk", 0.0), info.get("heading", 0.0) % 360.0,
            info.get("pos_x", 0.0), info.get("pos_z", 0.0)),
        "[SPACE] pause  [S] step  [D] dump JSON  [ESC/Q] quit",
        "[I] restart game   YOU are White: click a piece, then a square",
        "BOARD: drag pieces or click-select | right-drag to mark plans | promotion: Q/R/B/N",
        "viewport: MMB-drag orbit, wheel zoom",
    ]
    y = SCREEN_H - 20 * len(lines) - 8
    for ln in lines:
        surf.blit(f.render(ln, True, WHITE), (16, y))
        y += 20


# ============================================================================
# 5. ENGINE LOOP
# ============================================================================

def softmax(vec, temperature=1.0):
    v = np.asarray(vec, dtype=float)
    v = v - v.max()
    e = np.exp(v / max(1e-6, temperature))
    return e / e.sum()


class FlyChessAgent(object):
    """The connectome-driven player (always BLACK in single-agent mode)."""

    def __init__(self, gateway, name="fly"):
        self.net = ConnectomeNetwork(gateway)
        self.name = name
        self.da_reward = 0
        self.da_penalty = 0
        self.last_dopamine = 0.0
        self.last_move = None
        self.last_acts = None
        self.realistic = False      # fly-hardware mode: instincts, no search

    def choose_move(self, board, explore=True, think_seconds=0.0,
                    think_cb=None):
        """Sense -> deliberate -> select a legal move.

        With ``think_seconds > 0`` the fly runs time-bounded iterative-
        deepening negamax (material + piece-square tables, alpha-beta). Only
        a fully completed search depth can select a move; the connectome drive
        breaks exact search ties, so longer thinking does not leave unsearched
        moves with fake zero scores or randomly override the search.
        """
        legal = list(board.legal_moves)
        if not legal:
            return None
        body = getattr(self, "body", None)
        if body is not None and getattr(body, "eco", False):
            body = None        # ECO: skip proprioceptive salience computation
        sensory = self.net.encode_board(board, body=body)
        acts = self.net.propagate(sensory)
        motor = acts[MOTOR_REGION]
        drives = [float(motor[SQUARE_TO_IDX[m.from_square]]
                        + motor[64 + SQUARE_TO_IDX[m.to_square]])
                  for m in legal]

        if self.realistic:
            # REALISTIC MODE: the fly knows the rules but owns fly hardware -
            # no search tree, one reflex pass, choice made in fly-real time.
            scored = fly_instinct_move(board, motor)
            return scored[0][1], scored[0][0]

        search_scores = None
        if think_seconds and think_seconds > 0 and len(legal) > 1:
            search_scores, _search_depth = search_root_moves(
                board, legal, think_seconds, think_cb=think_cb)

        if search_scores is not None:
            # Do not let the exploration softmax undo the search result. The
            # connectome only breaks exact search ties, so a larger budget
            # cannot make the engine randomly ignore its completed search.
            best_score = max(search_scores)
            best_moves = [i for i, score in enumerate(search_scores)
                          if score == best_score]
            i = max(best_moves, key=lambda index: drives[index])
        elif think_seconds and think_seconds > 0:
            # If even the first iteration times out, return the strongest
            # connectome-drive move deterministically instead of adding noise.
            i = int(np.argmax(drives))
        elif explore and len(legal) > 1:
            probs = softmax(drives, TEMPERATURE)
            i = int(np.random.choice(len(legal), p=probs))
        else:
            i = int(np.argmax(drives))
        scores = (search_scores if search_scores is not None else drives)
        mv = legal[i]
        self.last_move = mv
        self.last_acts = (drives[i], 0.0, scores[i])
        return mv, scores[i]

    def learn(self, board_before, move, board_after):
        """Apply the dopamine/Hebbian rule for the move just played."""
        mover = board_before.turn
        r = move_reward(board_before, move)
        r += blunder_penalty(board_before, board_after, mover)
        fi = SQUARE_TO_IDX[move.from_square]
        ti = SQUARE_TO_IDX[move.to_square]
        score = self.last_acts[2] if self.last_acts else 0.0
        da = self.net.apply_dopamine(score, fi, ti, r)
        self.last_dopamine = da
        if r > 0:
            self.da_reward += 1
        elif r < 0:
            self.da_penalty += 1
        return r


def random_agent_move(board, rng):
    return rng.choice(list(board.legal_moves))


def run_match(agent, games, ply_limit, speed, selfplay, viz, rng,
              status_cb, interactive=False, think=THINK_TIME,
              realistic=False, animation_seconds=0.24,
              return_to_menu=False):
    """Play full games; visualize and learn in real time.

    interactive=True: the human plays White by clicking the on-screen
    chessboard (right half of the window); the connectome answers as Black
    and visibly fires its legs/wings with every move it makes.
    """
    fly_score = 0.0
    for game_no in range(1, games + 1):
        board = chess.Board()
        viz.board_panel.reset()
        viz.board_state = board.copy()
        clocks = {chess.WHITE: 300.0, chess.BLACK: 300.0}
        turn_started = time.time()
        ply = 0
        aborted = False
        fly_move_sq = None          # fly's last move, purple-highlighted

        def info_for(status=None, game_over=None):
            mode = ("YOU (White) vs CONNECTOME (Black)" if interactive else
                    "SELF-PLAY" if selfplay else
                    "fly(BLACK) vs random(WHITE)")
            if realistic:
                mode = "REALISTIC FLY-BRAIN: " + mode
            return {        "mode": mode,

                    "neuprint": "LIVE" if agent.net.gateway.live else "MOCK",
                    "dataset": NEUPRINT_DATASET,
                    "game": game_no, "games": games, "ply": ply,
                    "last_move": board.peek().uci() if board.move_stack else "-",
                    "last_chess_move": board.peek() if board.move_stack else None,
                    "fly_score": fly_score,
                    "da_reward": agent.da_reward,
                    "da_penalty": agent.da_penalty,
                    "mean_w": agent.net.mean_weight(),
                    "da_last": agent.last_dopamine,
                    "interactive": interactive,
                    "status": status, "game_over": game_over,
                    "eco": viz.body.eco,
                    "walk": viz.body.walk_cmd, "heading": viz.body.heading,
                    "pos_x": viz.body.pos.x, "pos_z": viz.body.pos.z,
                    "clocks": {"white": clocks[chess.WHITE] -
                               (time.time() - turn_started if board.turn == chess.WHITE else 0),
                               "black": clocks[chess.BLACK] -
                               (time.time() - turn_started if board.turn == chess.BLACK else 0)}}

        while not board.is_game_over(claim_draw=True) and ply < ply_limit:
            handle_input(viz)
            if viz.restart_request:
                viz.restart_request = False
                aborted = True
                break
            # paused: idle-render until SPACE resumes or [S] steps one ply
            while (viz.enabled and viz.paused and not viz.step_request
                   and not viz.restart_request):
                handle_input(viz)
                if viz.dump_request:
                    print("[dump] connectome snapshot -> %s"
                          % agent.net.dump_connectome())
                    viz.dump_request = False
                time.sleep(0)
            if viz.restart_request:
                viz.restart_request = False
                aborted = True
                break
            viz.step_request = False
            board_before = board.copy()
            turn_started = time.time()
            fly_moves = selfplay or board.turn == chess.BLACK

            if fly_moves:
                # BIOLOGY: the connectome drives the Black pieces, and like a
                # real animal it pauses to deliberate before committing to a
                # motor program.
                if realistic:
                    # REALISTIC MODE: no deliberation pause - the fly sees,
                    # decides, and moves in a fraction of a second.
                    result = agent.choose_move(board)
                    time.sleep(FLY_REACTION_TIME)   # photoreceptor/VNC latency
                elif think > 0 and len(list(board.legal_moves)) > 1:
                    viz.board_state = board.copy()
                    viz.body.hold = True      # stand still and think: the
                    # reflex layer freezes so no 'intellect' goes to the body
                    viz.render(agent.net.last_sensory, agent.net.last_motor,
                               info=info_for("thinking... (%0.0fs pause)" % think),
                               fly_move=fly_move_sq)

                    def think_tick(elapsed, depth):
                        # keep the window alive during deliberation; the body
                        # stays held, so no reflex work happens either
                        if not viz.enabled:
                            return
                        pygame.event.pump()
                        viz.render(agent.net.last_sensory,
                                   agent.net.last_motor,
                                   info=info_for("thinking... %.1fs (depth %d)"
                                                 % (elapsed, depth)),
                                   fly_move=fly_move_sq)

                    result = agent.choose_move(board,
                                               think_seconds=think,
                                               think_cb=think_tick)
                    viz.body.hold = False     # reflexes re-enabled
                else:
                    result = agent.choose_move(board)
                if result is None:
                    break
                mv, _total = result
                clocks[board_before.turn] -= time.time() - turn_started
                board.push(mv)
                viz.board_panel.record_move(board_before, mv)
                viz.board_panel.begin_move_animation(
                    board_before, board, mv, animation_seconds)
                r = agent.learn(board_before, mv, board.copy())
                fly_score += r
                if r > 0.5:
                    viz.body.trigger_reward("+DOPAMINE  %+.2f" % r)
                elif r < -0.05:
                    viz.body.trigger_penalty()
                fly_move_sq = mv
                status_cb(game_no, games, ply, mv.uci(), fly_score,
                          agent.da_reward, agent.da_penalty,
                          agent.last_dopamine)
            elif interactive:
                # HUMAN MOVE: idle-render while waiting for board clicks.
                # Reflex layer free: the fly may wander while you think.
                viz.body.hold = False
                mv = None
                while mv is None:
                    handle_input(viz)
                    if viz.board_panel.pending_move is not None:
                        mv = viz.board_panel.pending_move
                        viz.board_panel.pending_move = None
                    if viz.restart_request:
                        viz.restart_request = False
                        aborted = True
                        break
                    if viz.board_events:
                        event, pos = viz.board_events.popleft()
                        if event == "drop":
                            mv = viz.board_panel.end_drag(board, pos)
                        elif event == "arrow":
                            viz.board_panel.end_arrow(pos)
                    if mv is None and viz.board_clicks:
                        mv = viz.board_panel.handle_click(
                            board, viz.board_clicks.popleft())
                    viz.render(agent.net.last_sensory, agent.net.last_motor,
                               info=info_for("your move - you are White"),
                               fly_move=fly_move_sq)
                if aborted:
                    break
                clocks[board_before.turn] -= time.time() - turn_started
                board.push(mv)
                viz.board_panel.record_move(board_before, mv)
                viz.board_panel.begin_move_animation(
                    board_before, board, mv, animation_seconds)
                fly_move_sq = None
            else:
                # opponent: random legal moves (a 'control animal')
                viz.body.hold = False
                mv = random_agent_move(board, rng)
                clocks[board_before.turn] -= time.time() - turn_started
                board.push(mv)
                viz.board_panel.record_move(board_before, mv)
                viz.board_panel.begin_move_animation(
                    board_before, board, mv, animation_seconds)
                fly_move_sq = None
                status_cb(game_no, games, ply, mv.uci(), fly_score,
                          agent.da_reward, agent.da_penalty,
                          agent.last_dopamine)
            ply += 1
            viz.board_state = board.copy()
            viz.render(agent.net.last_sensory, agent.net.last_motor,
                       info=info_for(), fly_move=fly_move_sq)
            if viz.enabled and animation_seconds > 0:
                while viz.board_panel.animation_active():
                    handle_input(viz)
                    viz.render(agent.net.last_sensory, agent.net.last_motor,
                               info=info_for(), fly_move=fly_move_sq)
            if speed:
                time.sleep(1.0 / speed)
            if viz.dump_request:
                print("[dump] connectome snapshot -> %s"
                      % agent.net.dump_connectome())
                viz.dump_request = False
        replay_mode = "selfplay" if selfplay else ("play" if interactive else "watch")
        try:
            replay_path = save_pgn_replay(board, game_no, replay_mode, fly_score)
            print("[replay] saved -> %s" % replay_path)
        except OSError as exc:
            print("[replay] could not save game: %s" % exc)
        if aborted:
            break

        # ---------------- game over: reward bookkeeping ---------------------
        outcome = board.outcome(claim_draw=True)
        winner = outcome.winner if outcome else None
        if winner == chess.BLACK:
            fly_score += 25.0
            viz.body.trigger_reward("MATE!  BIG DA BURST")
            r = MATE_REWARD
        elif winner is None:
            r = 0.0
        else:
            viz.body.trigger_penalty()
            r = -MATE_REWARD
        agent.net.apply_dopamine(0.0, 0, 0, r)

        if interactive:
            if winner == chess.BLACK:
                status, over = "THE FLY WINS - checkmate", "fly"
            elif winner == chess.WHITE:
                status, over = "YOU WIN - dopamine crash for the fly", "human"
            else:
                status, over = ("Draw - %s"
                                % (outcome.termination if outcome
                                   else "ply limit"), "draw")
            print("Game %d over: %s (%d plies)" % (game_no, status, ply))
            # hold the final board on screen until SPACE (new game) / ESC
            viz.board_state = board.copy()
            if return_to_menu:
                viz.paused = False
                viz.render(agent.net.last_sensory, agent.net.last_motor,
                           info=info_for(status, over))
            else:
                viz.paused = True
                while viz.enabled and viz.paused and not viz.restart_request:
                    handle_input(viz)
                    viz.render(agent.net.last_sensory, agent.net.last_motor,
                               info=info_for(status, over))
                    time.sleep(0)
                viz.paused = False
                viz.restart_request = False
        else:
            viz.restart_request = False
            print("Game %d over: %s (%d plies)  running fly score %.1f"
                  % (game_no, outcome.termination if outcome else "ply limit",
                     ply, fly_score))
        if return_to_menu:
            break
    return fly_score


def handle_input(viz):
    if not viz.enabled:
        return
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if viz.screen_state != "playing":
            viz.handle_menu_event(ev)
            continue
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE and viz.menu_session_active:
                viz.restart_request = True
                continue
            if viz.board_panel.promotion_pending is not None:
                viz.board_panel.handle_key(ev.key)
            elif ev.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                sys.exit(0)
            elif ev.key == pygame.K_SPACE:
                viz.paused = not viz.paused
            elif ev.key == pygame.K_s:
                viz.step_request = True
            elif ev.key == pygame.K_d:
                viz.dump_request = True
            elif ev.key == pygame.K_i:
                viz.restart_request = True
        # ---- mouse: chessboard clicks (right pane) + Blender orbit (left) --
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button in (1, 3) and (viz.eco_only or ev.pos[0] >= SPLIT_X):
                if not viz.board_panel.begin_drag(viz.board_state, ev.pos,
                                                  button=ev.button):
                    if ev.button == 1:
                        viz.board_clicks.append(ev.pos)
            elif ev.button == 2 and ev.pos[0] < SPLIT_X:
                viz.mmb_drag = True                  # MMB: start orbiting
                viz.last_mouse = ev.pos
            elif ev.button == 4 and ev.pos[0] < SPLIT_X:
                viz.cam.zoom(0.9)                    # wheel up: zoom in
            elif ev.button == 5 and ev.pos[0] < SPLIT_X:
                viz.cam.zoom(1.1)                    # wheel down: zoom out
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and (viz.eco_only or ev.pos[0] >= SPLIT_X):
            # A release completes a drag only when a piece was picked up.
            # Click-to-select destinations are already queued on mouse-down;
            # treating their release as another click leaves stale duplicate
            # clicks waiting for the following turn.
            if viz.board_panel.drag_from is not None:
                viz.board_events.append(("drop", ev.pos))
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3 and (viz.eco_only or ev.pos[0] >= SPLIT_X):
            # Right-button drag arrows are independent of chess turn and are
            # committed directly on release. This avoids routing them through
            # the human move queue, which previously made them appear inert.
            viz.board_panel.end_arrow(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 2:
            viz.mmb_drag = False                     # release: stop orbiting
            viz.last_mouse = None
        elif ev.type == pygame.MOUSEMOTION:
            board_area = viz.eco_only or ev.pos[0] >= SPLIT_X
            if board_area and (viz.board_panel.drag_from is not None
                               or viz.board_panel.arrow_from is not None):
                # Both piece drags and right-button analysis arrows are updated
                # on every motion event, so the cursor ghost/arrow never lags.
                viz.board_panel.drag_to(ev.pos)
            elif viz.mmb_drag:
                viz.cam.orbit(ev.rel[0], ev.rel[1])


class Visualizer(object):
    """Owns the pygame window, camera, fly body, and the render tick."""

    def __init__(self, enabled=True):
        self.enabled = enabled and pygame is not None
        # BoardPanel owns cached font objects even in headless mode. Initialize
        # the font subsystem before constructing it, while only opening a
        # window when GUI rendering was requested.
        if pygame is not None and not pygame.font.get_init():
            pygame.font.init()
        self.eco_only = False
        self.paused = False
        self.step_request = False
        self.dump_request = False
        self.restart_request = False
        self.mmb_drag = False                    # Blender-style MMB orbit state
        self.last_mouse = None
        self.body = FlyBody()
        self.cam = Camera()
        self.screen_state = "menu"
        self.last_settings = load_settings()
        self._menu_font_cache = {}
        self.menu_buttons = {}
        self.menu_hover = None
        self.menu_action = None
        self.menu_slider_rect = pygame.Rect(0, 0, 0, 0)
        self.difficulty_dragging = False
        self.training_mode = False
        self.replay_paths = []
        self.replay_entries = []
        self.replay_scroll = 0
        self.replay_game = None
        self.replay_board = None
        self.replay_moves = []
        self.replay_san_moves = []
        self.replay_index = 0
        self.replay_selected = 0
        self.replay_playing = False
        self.replay_last_step = 0.0
        self.review_analysis = {}
        self.review_analysis_queue = queue.Queue()
        self.review_analysis_thread = None
        self.review_analysis_stop = None
        self.review_analysis_generation = 0
        self.review_analysis_running = False
        self.review_analysis_complete = False
        self.review_analysis_total = 0
        self.review_analysis_done = 0
        self.review_analysis_error = ""
        self.menu_session_active = False
        self.board_panel = BoardPanel()          # right-half chessboard
        self.board_state = chess.Board()         # latest state to draw
        self.board_clicks = deque()              # legacy click queue
        self.board_events = deque()              # drag lifecycle events
        self.last_info = {}
        self.last_sensory = None
        self.last_motor = None
        self.last_frame_time = time.perf_counter()
        if self.enabled:
            pygame.init()
            pygame.display.set_caption("Fruit Fly Connectome Chess Bot")
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()

    def render(self, sensory, motor, info, fly_move=None):
        self.last_sensory, self.last_motor, self.last_info = sensory, motor, info
        if not self.enabled:
            return
        if self.screen_state in ("menu", "settings", "training", "replays", "review"):
            self.clock.tick_busy_loop(TARGET_FPS)
            if self.screen_state == "menu":
                self.draw_menu()
            elif self.screen_state == "settings":
                self.draw_settings()
            elif self.screen_state == "training":
                self.draw_training()
            else:
                self.draw_replays()
            pygame.display.flip()
            return
        # tick_busy_loop keeps input/render cadence tight instead of allowing
        # a slow OS timer to make drag motion feel like a 10 FPS app.
        dt = self.clock.tick_busy_loop(TARGET_FPS) / 1000.0
        self.body.time += dt
        self.cam.follow(self.body.pos)     # keep the walking fly framed
        self.cam.tick(dt, self.mmb_drag)   # no auto-rotate; MMB orbits only
        if not self.eco_only:
            self.body.apply_motor(motor, dt)

        if self.eco_only:
            # ECO is a chess-training view: no fly viewport, no 3D grid, and
            # no motor animation work. The board uses the entire window.
            self.screen.fill((28, 28, 34))
            self._draw_board_panel(info, fly_move, full_screen=True)
        else:
            # ---- left pane: 3D viewport ------------------------------------
            self.screen.fill((12, 12, 20))
            self._grid()
            if self.body.flash_timer > 0:
                k = self.body.flash_timer / 0.6
                wash = pygame.Surface((SPLIT_X, SCREEN_H), pygame.SRCALPHA)
                wash.fill((0, int(200 * k), 60, int(70 * k)))
                self.screen.blit(wash, (0, 0))
            self.body.draw(self.screen, self.cam, sensory)
            # ---- right pane: dedicated chessboard pane --------------------
            self.screen.fill((28, 28, 34),
                             rect=pygame.Rect(SPLIT_X, 0,
                                              SCREEN_W - SPLIT_X, SCREEN_H))
            pygame.draw.line(self.screen, (70, 70, 85), (SPLIT_X, 0),
                             (SPLIT_X, SCREEN_H), 3)
            self._draw_viewport_hint()
            self._draw_board_panel(info, fly_move)
        draw_hud(self.screen, info)
        pygame.display.flip()

    def _menu_font(self, size, bold=False):
        cache = getattr(self, "_menu_font_cache", None)
        if cache is None:
            cache = self._menu_font_cache = {}
        key = (size, bold)
        if key not in cache:
            cache[key] = pygame.font.SysFont("segoeui,arial", size, bold=bold)
        return cache[key]

    def _text(self, text, pos, size=22, color=(225, 235, 248), bold=False):
        self.screen.blit(self._menu_font(size, bold).render(text, True, color), pos)

    def _button(self, key, label, rect, accent=False, small=False):
        rect = pygame.Rect(rect)
        self.menu_buttons[key] = rect
        hovered = self.menu_hover == key
        fill = (35, 49, 67) if not hovered else (48, 68, 91)
        if accent:
            fill = (18, 132, 154) if not hovered else (24, 162, 179)
        pygame.draw.rect(self.screen, fill, rect, border_radius=13)
        pygame.draw.rect(self.screen, (61, 82, 104) if not accent else (69, 213, 219),
                         rect, 2, border_radius=13)
        font_size = 19 if small else 23
        text = self._menu_font(font_size, True).render(label, True, (242, 248, 255))
        self.screen.blit(text, text.get_rect(center=rect.center))
        return rect

    def _draw_page_backdrop(self, eyebrow, title, subtitle=""):
        self.screen.fill((10, 15, 27))
        pygame.draw.rect(self.screen, (16, 23, 38), (36, 32, SCREEN_W - 72,
                         SCREEN_H - 64), border_radius=28)
        pygame.draw.rect(self.screen, (36, 56, 77), (36, 32, SCREEN_W - 72,
                         SCREEN_H - 64), 2, border_radius=28)
        self._text(eyebrow.upper(), (88, 66), 16, (72, 210, 218), True)
        self._text(title, (84, 98), 43, (240, 246, 255), True)
        if subtitle:
            self._text(subtitle, (88, 155), 18, (151, 169, 191))

    def draw_menu(self):
        self._draw_page_backdrop("Janelia connectome project", "Fruit Fly Chess",
                                 "A neural-connectome chess experiment")
        self.menu_buttons = {}
        # Main actions on the left; difficulty is deliberately adjacent to Play.
        play_rect = self._button("play", "PLAY", (112, 280, 330, 74), accent=True)
        slider = pygame.Rect(496, 314, 440, 8)
        self.menu_slider_rect = slider
        self._text("DIFFICULTY", (496, 277), 15, (151, 169, 191), True)
        pygame.draw.rect(self.screen, (48, 62, 79), slider, border_radius=6)
        value = int(self.last_settings["difficulty"])
        knob_x = slider.x + round((value - 1) / 9 * slider.w)
        pygame.draw.rect(self.screen, (69, 213, 219),
                         pygame.Rect(slider.x, slider.y, knob_x-slider.x, slider.h),
                         border_radius=6)
        pygame.draw.circle(self.screen, (234, 248, 255), (knob_x, slider.centery), 13)
        self._text("CASUAL", (slider.x, 340), 13, (132, 150, 172))
        self._text("DEEP THINKING", (slider.right - 135, 340), 13, (132, 150, 172))
        self._text("LEVEL %d" % value, (slider.x + slider.w // 2 - 35, 364),
                   16, (100, 222, 226), True)
        self._button("training", "TRAINING", (112, 390, 824, 64))
        self._button("replays", "GAME REPLAYS", (112, 470, 824, 64))
        self._button("settings", "SETTINGS", (112, 550, 824, 64))
        self._button("quit", "EXIT", (112, 630, 824, 58), small=True)

        # A compact dashboard card gives the menu a clear, intentional identity.
        card = pygame.Rect(1032, 235, 690, 510)
        pygame.draw.rect(self.screen, (20, 31, 49), card, border_radius=22)
        pygame.draw.rect(self.screen, (43, 70, 91), card, 2, border_radius=22)
        self._text("THE FLY'S GAME", (1080, 278), 17, (74, 213, 219), True)
        self._text("64", (1080, 325), 52, (239, 245, 255), True)
        self._text("visual square channels", (1082, 385), 17, (164, 182, 204))
        self._text("15", (1312, 325), 52, (239, 245, 255), True)
        self._text("connectome regions", (1314, 385), 17, (164, 182, 204))
        self._text("192", (1530, 325), 52, (239, 245, 255), True)
        self._text("motor channels", (1532, 385), 17, (164, 182, 204))
        pygame.draw.line(self.screen, (49, 69, 89), (1080, 440), (1674, 440), 2)
        self._text("PLAY", (1080, 474), 15, (87, 210, 218), True)
        self._text("Choose a difficulty, then challenge the connectome.",
                   (1080, 503), 19, (219, 230, 243))
        self._text("TRAINING", (1080, 555), 15, (87, 210, 218), True)
        self._text("Run self-play games and keep the learned weights.",
                   (1080, 584), 19, (219, 230, 243))
        self._text("REPLAYS", (1080, 636), 15, (87, 210, 218), True)
        self._text("Analyze saved games with Stockfish and inspect each move.",
                   (1080, 665), 19, (219, 230, 243))
        self._text("Settings and replays are stored next to the app.",
                   (112, 738), 15, (125, 144, 167))

    def draw_settings(self):
        self._draw_page_backdrop("Preferences", "Settings",
                                 "These choices are saved locally for your next launch.")
        self.menu_buttons = {}
        rows = [
            ("eco_mode", "ECO BOARD", "Hide the fly viewport and use the full screen for chess."),
            ("realistic_mode", "REALISTIC MODE", "Use the fast instinct-only fly decision mode."),
            ("offline", "OFFLINE CONNECTOME", "Skip neuPrint and use the built-in mock network."),
        ]
        for i, (key, label, desc) in enumerate(rows):
            y = 250 + i * 118
            self._text(label, (150, y), 22, (231, 239, 249), True)
            self._text(desc, (150, y + 35), 17, (145, 164, 186))
            enabled = self.last_settings[key]
            self._button("toggle_" + key, "ON" if enabled else "OFF",
                         (1110, y - 7, 170, 55), accent=enabled, small=True)
        y = 605
        self._text("MOVE ANIMATION", (150, y), 22, (231, 239, 249), True)
        self._text("Duration: %.2f seconds" % self.last_settings["animation_seconds"],
                   (150, y + 35), 17, (145, 164, 186))
        self._button("anim_down", "−", (1110, y - 7, 72, 55), small=True)
        self._button("anim_up", "+", (1200, y - 7, 72, 55), small=True)

        engine_path = self.last_settings.get("stockfish_path", "")
        resolved_engine = resolve_stockfish_path(engine_path)
        engine_label = (resolved_engine or engine_path or
                        "Not configured — select your Stockfish executable")
        if len(engine_label) > 72:
            engine_label = "..." + engine_label[-69:]
        self._text("STOCKFISH REVIEW ENGINE", (150, 700), 22,
                   (231, 239, 249), True)
        self._text(engine_label, (150, 740), 16,
                   (107, 222, 183) if resolved_engine else (255, 190, 120))
        self._button("stockfish_browse", "CHOOSE EXE", (1110, 690, 180, 55),
                     accent=True, small=True)
        self._button("stockfish_auto", "AUTO-DETECT", (1310, 690, 180, 55),
                     small=True)
        self._button("stockfish_clear", "CLEAR", (1510, 690, 150, 55),
                     small=True)
        self._button("settings_back", "SAVE & BACK", (150, 850, 350, 62),
                     accent=True)

    def _choose_stockfish_path(self):
        """Open a native executable picker for the user's local Stockfish."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                parent=root, title="Select the Stockfish executable",
                filetypes=(("Stockfish executable", "*.exe"),
                           ("All files", "*.*")))
            root.destroy()
        except Exception as exc:
            self.review_analysis_error = "Could not open file picker: %s" % exc
            return
        if selected:
            self.last_settings["stockfish_path"] = selected
            self.review_analysis_error = ""

    def draw_training(self):
        self._draw_page_backdrop("Learn through self-play", "Training",
                                 "Games are recorded as PGN; learned weights are saved after the session.")
        self.menu_buttons = {}
        self._text("GAMES", (170, 310), 20, (230, 238, 248), True)
        self._text("%d self-play games" % self.last_settings["training_games"],
                   (170, 352), 24, (138, 220, 224))
        self._button("games_down", "−", (680, 325, 74, 60), small=True)
        self._button("games_up", "+", (772, 325, 74, 60), small=True)
        self._text("Difficulty level %d  ·  %.2fs per move" % (
            self.last_settings["difficulty"],
            difficulty_think_seconds(self.last_settings["difficulty"])),
            (170, 442), 20, (170, 187, 207))
        self._button("training_start", "START TRAINING", (170, 535, 390, 74), accent=True)
        self._button("training_back", "BACK", (170, 635, 390, 60))

    def refresh_replay_list(self):
        """Scan PGN headers once when opening the replay browser."""
        self.replay_paths = list_replays()
        self.replay_entries = []
        for path in self.replay_paths:
            entry = {"path": path, "white": "White", "black": "Black",
                     "result": "*", "mode": "game", "plies": 0, "date": "",
                     "saved_at": ""}
            try:
                entry["saved_at"] = datetime.fromtimestamp(
                    os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
                with open(path, "r", encoding="utf-8") as stream:
                    game = chess.pgn.read_game(stream)
                if game is not None:
                    entry.update({
                        "white": game.headers.get("White", "White"),
                        "black": game.headers.get("Black", "Black"),
                        "result": game.headers.get("Result", "*"),
                        "mode": game.headers.get("Mode", "game"),
                        "date": game.headers.get("Date", ""),
                        "plies": sum(1 for _ in game.mainline_moves()),
                    })
            except (OSError, ValueError):
                pass
            self.replay_entries.append(entry)
        max_scroll = max(0, len(self.replay_entries) - 10)
        self.replay_scroll = max(0, min(self.replay_scroll, max_scroll))
        self.replay_selected = max(0, min(self.replay_selected,
                                          max(0, len(self.replay_entries) - 1)))

    def _sync_review_history(self):
        """Rebuild captured pieces and SAN history for the visible replay ply."""
        if self.replay_game is None:
            return
        panel = self.board_panel
        panel.history = []
        panel.captured_white = []
        panel.captured_black = []
        board = self.replay_game.board()
        for move in self.replay_moves[:self.replay_index]:
            panel.record_move(board, move)
            board.push(move)

    def _load_replay(self, index):
        paths = self.replay_paths
        if not paths:
            return False
        index = max(0, min(len(paths) - 1, index))
        try:
            with open(paths[index], "r", encoding="utf-8") as stream:
                game = chess.pgn.read_game(stream)
            if game is None:
                return False
        except (OSError, ValueError):
            return False
        self.board_panel.reset()
        self.replay_selected = index
        self._cancel_review_analysis()
        self.replay_game = game
        self.replay_moves = list(game.mainline_moves())
        self.replay_san_moves = []
        san_board = game.board()
        for move in self.replay_moves:
            self.replay_san_moves.append(san_board.san(move))
            san_board.push(move)
        self.replay_index = 0
        self.replay_board = game.board()
        self.replay_last_move = None
        self.replay_playing = False
        self.review_analysis = {}
        self.review_analysis_error = ""
        self.review_analysis_complete = False
        self.review_analysis_total = 0
        self.review_analysis_done = 0
        self._sync_review_history()
        self.board_state = self.replay_board.copy()
        self.screen_state = "review"
        return True

    def _cancel_review_analysis(self):
        """Stop engine work for the current review without blocking the UI."""
        stop_event = self.review_analysis_stop
        if stop_event is not None:
            stop_event.set()
        self.review_analysis_generation += 1
        self.review_analysis_running = False
        self.review_analysis_complete = False

    def _start_stockfish_analysis(self):
        """Analyse each pre-move PGN position on a worker thread."""
        if self.replay_game is None or self.replay_analysis_busy():
            return
        if self.review_analysis_thread and self.review_analysis_thread.is_alive():
            self._cancel_review_analysis()
        engine_path = resolve_stockfish_path(
            self.last_settings.get("stockfish_path", ""))
        if not engine_path:
            self.review_analysis_error = (
                "Stockfish not found. Set its executable in Settings first.")
            return

        self.review_analysis_error = ""
        self.review_analysis = {}
        self.review_analysis_complete = False
        self.review_analysis_done = 0
        self.review_analysis_total = len(self.replay_moves)
        if self.review_analysis_total == 0:
            self.review_analysis_error = "This saved game has no moves to analyze."
            return
        self.review_analysis_generation += 1
        generation = self.review_analysis_generation
        stop_event = threading.Event()
        self.review_analysis_stop = stop_event
        self.review_analysis_running = True
        game_snapshot = self.replay_game
        moves_snapshot = list(self.replay_moves)
        output_queue = self.review_analysis_queue

        def worker():
            engine = None
            try:
                engine = chess.engine.SimpleEngine.popen_uci(
                    engine_path, timeout=10.0)
                board = game_snapshot.board()
                for ply, move in enumerate(moves_snapshot):
                    if stop_event.is_set():
                        break
                    before = board.copy()
                    mover_name = "White" if before.turn == chess.WHITE else "Black"
                    san = before.san(move)
                    info = engine.analyse(
                        before,
                        chess.engine.Limit(time=STOCKFISH_ANALYSIS_SECONDS))
                    if stop_event.is_set():
                        break
                    best_score = info["score"].pov(before.turn)
                    best_cp = best_score.score(mate_score=STOCKFISH_MATE_SCORE)
                    pv = info.get("pv", [])
                    best_line = before.variation_san(pv[:8]) if pv else ""
                    best_move_san = before.san(pv[0]) if pv else "unavailable"

                    after_played = before.copy()
                    after_played.push(move)
                    played_info = engine.analyse(
                        after_played,
                        chess.engine.Limit(time=STOCKFISH_ANALYSIS_SECONDS))
                    if stop_event.is_set():
                        break
                    # Compare the post-move position from the original
                    # mover's perspective, not the opponent-to-move POV.
                    played_score = played_info["score"].pov(before.turn)
                    played_cp = played_score.score(
                        mate_score=STOCKFISH_MATE_SCORE)
                    centipawn_loss = max(0, best_cp - played_cp)
                    quality, quality_color = classify_stockfish_move(
                        centipawn_loss,
                        pv[0].uci() if pv else "",
                        move.uci())
                    output_queue.put(("move", generation, ply, {
                        "player": mover_name,
                        "played": san,
                        "played_uci": move.uci(),
                        "best_evaluation": best_cp,
                        "best_mate": best_score.mate(),
                        "played_evaluation": played_cp,
                        "played_mate": played_score.mate(),
                        "centipawn_loss": centipawn_loss,
                        "quality": quality,
                        "quality_color": quality_color,
                        "best_move": pv[0].uci() if pv else "",
                        "best_move_san": best_move_san,
                        "best_line": best_line,
                        "depth": info.get("depth", 0),
                    }))
                    board.push(move)
                if not stop_event.is_set():
                    output_queue.put(("done", generation, None, None))
            except Exception as exc:
                output_queue.put(("error", generation, None,
                                  "Stockfish analysis failed: %s" % exc))
            finally:
                if engine is not None:
                    try:
                        engine.quit()
                    except Exception:
                        try:
                            engine.close()
                        except Exception:
                            pass

        self.review_analysis_thread = threading.Thread(
            target=worker, name="StockfishReview", daemon=True)
        self.review_analysis_thread.start()

    def replay_analysis_busy(self):
        """Drain worker messages on the UI thread and report current progress."""
        while True:
            try:
                kind, generation, ply, payload = self.review_analysis_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self.review_analysis_generation:
                continue
            if kind == "move":
                self.review_analysis[ply] = payload
                self.review_analysis_done = len(self.review_analysis)
            elif kind == "error":
                self.review_analysis_error = payload
                self.review_analysis_running = False
                self.review_analysis_complete = False
            elif kind == "done":
                self.review_analysis_running = False
                self.review_analysis_complete = True
        return self.review_analysis_running

    def _replay_step(self, direction):
        if self.replay_game is None or self.replay_board is None:
            return
        new_index = max(0, min(len(self.replay_moves),
                               self.replay_index + direction))
        if new_index == self.replay_index:
            self.replay_playing = False
            return
        if new_index == self.replay_index + 1:
            before = self.replay_board.copy()
            move = self.replay_moves[self.replay_index]
            self.replay_board.push(move)
            self.board_panel.begin_move_animation(
                before, self.replay_board, move,
                self.last_settings["animation_seconds"])
            self.replay_last_move = move
        else:
            self.replay_board = self.replay_game.board()
            for move in self.replay_moves[:new_index]:
                self.replay_board.push(move)
            self.replay_last_move = (self.replay_moves[new_index - 1]
                                     if new_index else None)
            self.board_panel.animation_segments = []
            self.board_panel.animation_source = None
            self.board_panel.animation_capture_square = None
        self.replay_index = new_index
        self.board_state = self.replay_board.copy()
        self._sync_review_history()

    def draw_replays(self):
        self.replay_analysis_busy()
        if self.screen_state == "replays":
            self._draw_page_backdrop("Saved PGN files", "Game replays",
                                     "Select a saved game to step through or autoplay.")
            self.menu_buttons = {}
            self._button("replay_back", "BACK", (96, 880, 240, 58))
            if not self.replay_entries:
                self._text("No saved games yet. Play a game or run training first.",
                           (120, 250), 24, (172, 190, 210))
                return
            start_y = 220
            visible = self.replay_entries[self.replay_scroll:self.replay_scroll + 10]
            for row_index, entry in enumerate(visible):
                index = self.replay_scroll + row_index
                row = pygame.Rect(100, start_y + row_index * 65, 1450, 52)
                selected = index == self.replay_selected
                pygame.draw.rect(self.screen, (31, 51, 69) if selected else (19, 31, 47),
                                 row, border_radius=10)
                pygame.draw.rect(self.screen, (57, 150, 163) if selected else (40, 60, 79),
                                 row, 2, border_radius=10)
                saved_at = entry["saved_at"] or entry["date"]
                mode_label = {"play": "PLAY", "selfplay": "SELF-PLAY",
                              "watch": "WATCH"}.get(entry["mode"], entry["mode"].upper())
                label = ("%s   %s vs %s   %s   %d plies   %s" % (
                    saved_at, entry["white"], entry["black"], entry["result"],
                    entry["plies"], mode_label))
                self._text(label, (row.x + 20, row.y + 13), 17,
                           (231, 239, 248), selected)
                self.menu_buttons["replay_row_%d" % index] = row
            self._button("replay_older", "OLDER ↓", (420, 880, 190, 58), small=True)
            self._button("replay_newer", "NEWER ↑", (630, 880, 190, 58), small=True)
            self._text("Showing %d-%d of %d  ·  ↑/↓ or mouse wheel to scroll" % (
                self.replay_scroll + 1,
                min(self.replay_scroll + len(visible), len(self.replay_entries)),
                len(self.replay_entries)), (860, 898), 15, (139, 159, 181))
            return

        if self.replay_playing and time.perf_counter() >= self.replay_last_step + 0.65:
            self._replay_step(1)
            self.replay_last_step = time.perf_counter()
        self.screen.fill((10, 15, 27))
        pygame.draw.rect(self.screen, (16, 23, 38), (36, 32, SCREEN_W - 72,
                         SCREEN_H - 64), border_radius=28)
        self._text("GAME REVIEW", (72, 65), 15, (73, 212, 219), True)
        name = (self.replay_game.headers.get("Event", "Saved game")
                if self.replay_game else "Saved game")
        self._text(name, (72, 93), 32, (238, 245, 255), True)
        self._text("Move %d / %d" % (self.replay_index, len(self.replay_moves)),
                   (72, 155), 19, (155, 174, 196))
        if self.replay_game:
            self._text("%s  vs  %s" % (self.replay_game.headers.get("White", "White"),
                                       self.replay_game.headers.get("Black", "Black")),
                       (72, 188), 17, (155, 174, 196))
        self.menu_buttons = {}
        self._button("review_back", "BACK TO LIST", (72, 850, 250, 58))
        self._button("review_prev", "◀ PREV", (72, 545, 150, 58), small=True)
        self._button("review_play", "PAUSE" if self.replay_playing else "PLAY",
                     (242, 545, 150, 58), accent=True, small=True)
        self._button("review_next", "NEXT ▶", (412, 545, 150, 58), small=True)
        self._button("review_start", "|◀", (72, 625, 105, 50), small=True)
        self._button("review_end", "▶|", (197, 625, 105, 50), small=True)
        if self.review_analysis_running:
            self._button("review_analysis", "ANALYZING...", (72, 455, 310, 58), small=True)
            self._button("review_analysis_stop", "STOP", (392, 455, 115, 58), small=True)
        elif self.review_analysis_complete:
            self._button("review_analysis", "ANALYSIS READY · RUN AGAIN", (72, 455, 310, 58), accent=True, small=True)
        else:
            self._button("review_analysis", "ANALYZE WITH STOCKFISH", (72, 455, 310, 58), accent=True, small=True)
        ply_to_show = self.replay_index - 1
        current_analysis = self.review_analysis.get(ply_to_show)
        if current_analysis:
            def fmt_score(cp, mate):
                if mate is not None:
                    return "M+%d" % mate if mate > 0 else "M%d" % mate
                return "%+.2f" % (cp / 100.0)
            best_text = fmt_score(current_analysis["best_evaluation"],
                                  current_analysis["best_mate"])
            played_text = fmt_score(current_analysis["played_evaluation"],
                                    current_analysis["played_mate"])
            loss_cp = current_analysis["centipawn_loss"]
            loss = loss_cp / 100.0
            best_move_text = current_analysis.get(
                "best_move_san", current_analysis["best_move"] or "unavailable")
            self._text("Best %s  ·  after played move %s  ·  loss %.2f" % (
                best_text, played_text, loss), (72, 255), 18,
                (107, 222, 183), True)
            self._text("Played %s: %s  |  best: %s  |  depth %s" % (
                current_analysis["player"], current_analysis["played"],
                best_move_text, current_analysis["depth"]),
                (72, 292), 16, (220, 231, 244))
            self._text(("Line: " + current_analysis["best_line"])
                       if current_analysis["best_line"] else "Best line unavailable",
                       (72, 322), 15, (160, 182, 207))
            quality = current_analysis.get("quality", "Good")
            quality_color = current_analysis.get(
                "quality_color", STOCKFISH_MOVE_QUALITY_COLORS.get(
                    quality, STOCKFISH_MOVE_QUALITY_COLORS["Good"]))
            quality_font = self._menu_font(15, True)
            quality_text = quality_font.render(quality.upper(), True,
                                               (17, 27, 35))
            quality_badge = pygame.Rect(72, 354,
                                       quality_text.get_width() + 20, 31)
            pygame.draw.rect(self.screen, quality_color, quality_badge,
                             border_radius=8)
            self.screen.blit(quality_text,
                             (quality_badge.x + 10, quality_badge.y + 7))
            self._text("%d cp lost" % loss_cp,
                       (quality_badge.right + 12, quality_badge.y + 7),
                       15, (193, 207, 223))
        elif self.review_analysis_running:
            self._text("Analyzing saved game: %d / %d moves" % (
                self.review_analysis_done, self.review_analysis_total),
                (72, 260), 18, (160, 182, 207))
        elif self.review_analysis_error:
            self._text(self.review_analysis_error[:70], (72, 260),
                       17, (255, 166, 120))
        elif self.review_analysis:
            self._text("No Stockfish result for this ply yet.", (72, 260),
                       17, (160, 182, 207))
        else:
            self._text("Choose ANALYZE to score every move in this PGN.",
                       (72, 260), 17, (160, 182, 207))
        self._text("Move grades use centipawn loss vs. Stockfish's top move:",
                   (72, 397), 14, (139, 159, 181))
        self._text("Best <=10 cp  |  Excellent <=30  |  Good <=100 cp",
                   (72, 416), 14, (174, 191, 210))
        self._text("Inaccuracy <=200 cp  |  Mistake <=400  |  Blunder >400 cp",
                   (72, 435), 14, (174, 191, 210))
        self.board_panel.x0, self.board_panel.y0 = 1000, 132
        self.board_panel.draw(self.screen, self.replay_board or chess.Board(),
                              last_move=self.replay_last_move, review=True,
                              review_analysis=self.review_analysis,
                              review_san_moves=self.replay_san_moves,
                              review_index=self.replay_index)
        self._text("← / → step  ·  SPACE play/pause  ·  A analyze  ·  HOME/END jump  ·  ESC back",
                   (72, 730), 13, (119, 141, 164))

    def handle_menu_event(self, ev):
        if ev.type == pygame.MOUSEWHEEL and self.screen_state == "replays":
            self.replay_scroll = max(0, min(
                max(0, len(self.replay_entries) - 10), self.replay_scroll - ev.y))
            if self.replay_entries:
                self.replay_selected = max(self.replay_scroll, min(
                    self.replay_selected,
                    min(len(self.replay_entries) - 1, self.replay_scroll + 9)))
            return
        if ev.type == pygame.MOUSEMOTION:
            self.menu_hover = next((key for key, rect in self.menu_buttons.items()
                                    if rect.collidepoint(ev.pos)), None)
            if self.difficulty_dragging and self.screen_state == "menu":
                self._set_difficulty_from_x(ev.pos[0])
            return
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.difficulty_dragging:
                save_settings(self.last_settings)
            self.difficulty_dragging = False
            return
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                if self.screen_state in ("settings", "training", "replays", "review"):
                    if self.screen_state == "review":
                        self._cancel_review_analysis()
                        self.screen_state = "replays"
                    else:
                        self.screen_state = "menu"
                    self.replay_playing = False
                    save_settings(self.last_settings)
                elif self.screen_state == "menu":
                    self.screen_state = "quit"
                return
            if self.screen_state == "review":
                if ev.key == pygame.K_LEFT:
                    self.replay_playing = False
                    self._replay_step(-1)
                elif ev.key == pygame.K_RIGHT:
                    self.replay_playing = False
                    self._replay_step(1)
                elif ev.key == pygame.K_SPACE:
                    self.replay_playing = not self.replay_playing
                    self.replay_last_step = time.perf_counter()
                elif ev.key == pygame.K_HOME:
                    self.replay_playing = False
                    self.replay_index = 0
                    self.replay_board = self.replay_game.board()
                    self.replay_last_move = None
                    self.board_panel.animation_segments = []
                    self.board_panel.animation_source = None
                    self.board_panel.animation_capture_square = None
                    self.board_state = self.replay_board.copy()
                    self._sync_review_history()
                elif ev.key == pygame.K_END:
                    self.replay_playing = False
                    while self.replay_index < len(self.replay_moves):
                        self._replay_step(1)
                elif ev.key == pygame.K_a:
                    if self.review_analysis_running:
                        self._cancel_review_analysis()
                    else:
                        self._start_stockfish_analysis()
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.replay_playing = not self.replay_playing
                    self.replay_last_step = time.perf_counter()
            elif self.screen_state == "replays":
                if ev.key in (pygame.K_UP, pygame.K_DOWN) and self.replay_entries:
                    delta = -1 if ev.key == pygame.K_UP else 1
                    self.replay_selected = max(0, min(
                        len(self.replay_entries) - 1,
                        self.replay_selected + delta))
                    if self.replay_selected < self.replay_scroll:
                        self.replay_scroll = self.replay_selected
                    elif self.replay_selected >= self.replay_scroll + 10:
                        self.replay_scroll = self.replay_selected - 9
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self._load_replay(self.replay_selected)
            return
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return
        pos = ev.pos
        if self.screen_state == "menu":
            if self.menu_slider_rect.inflate(0, 28).collidepoint(pos):
                self.difficulty_dragging = True
                self._set_difficulty_from_x(pos[0])
                return
            for key, rect in self.menu_buttons.items():
                if not rect.collidepoint(pos):
                    continue
                if key == "play":
                    self.menu_action = "play"
                    save_settings(self.last_settings)
                    self.screen_state = "playing"
                elif key == "training":
                    self.screen_state = "training"
                elif key == "replays":
                    self.refresh_replay_list()
                    self.screen_state = "replays"
                elif key == "settings":
                    self.screen_state = "settings"
                elif key == "quit":
                    self.screen_state = "quit"
                break
        elif self.screen_state == "settings":
            for key, rect in self.menu_buttons.items():
                if not rect.collidepoint(pos):
                    continue
                toggle = key[len("toggle_"):] if key.startswith("toggle_") else None
                if toggle:
                    self.last_settings[toggle] = not self.last_settings[toggle]
                elif key == "anim_down":
                    self.last_settings["animation_seconds"] = max(
                        0.08, self.last_settings["animation_seconds"] - 0.04)
                elif key == "anim_up":
                    self.last_settings["animation_seconds"] = min(
                        0.8, self.last_settings["animation_seconds"] + 0.04)
                elif key == "stockfish_browse":
                    self._choose_stockfish_path()
                elif key == "stockfish_auto":
                    detected = resolve_stockfish_path()
                    self.last_settings["stockfish_path"] = detected or ""
                    self.review_analysis_error = ("" if detected else
                        "Stockfish was not found on PATH; choose its executable.")
                elif key == "stockfish_clear":
                    self.last_settings["stockfish_path"] = ""
                    self.review_analysis_error = ""
                elif key == "settings_back":
                    save_settings(self.last_settings)
                    self.screen_state = "menu"
                break
        elif self.screen_state == "training":
            for key, rect in self.menu_buttons.items():
                if not rect.collidepoint(pos):
                    continue
                if key == "games_down":
                    self.last_settings["training_games"] = max(
                        1, self.last_settings["training_games"] - 1)
                elif key == "games_up":
                    self.last_settings["training_games"] = min(
                        50, self.last_settings["training_games"] + 1)
                elif key == "training_start":
                    self.menu_action = "training"
                    save_settings(self.last_settings)
                    self.screen_state = "playing"
                elif key == "training_back":
                    self.screen_state = "menu"
                break
        elif self.screen_state == "replays":
            for key, rect in self.menu_buttons.items():
                if not rect.collidepoint(pos):
                    continue
                if key == "replay_back":
                    self.screen_state = "menu"
                elif key == "replay_older":
                    self.replay_scroll = min(
                        max(0, len(self.replay_entries) - 10),
                        self.replay_scroll + 10)
                    self.replay_selected = self.replay_scroll
                elif key == "replay_newer":
                    self.replay_scroll = max(0, self.replay_scroll - 10)
                    self.replay_selected = self.replay_scroll
                elif key.startswith("replay_row_"):
                    self._load_replay(int(key.rsplit("_", 1)[1]))
                break
        elif self.screen_state == "review":
            for key, rect in self.menu_buttons.items():
                if not rect.collidepoint(pos):
                    continue
                if key == "review_back":
                    self.replay_playing = False
                    self._cancel_review_analysis()
                    self.screen_state = "replays"
                elif key == "review_analysis":
                    if not self.review_analysis_running:
                        self._start_stockfish_analysis()
                elif key == "review_analysis_stop":
                    self._cancel_review_analysis()
                elif key == "review_prev":
                    self.replay_playing = False
                    self._replay_step(-1)
                elif key == "review_next":
                    self.replay_playing = False
                    self._replay_step(1)
                elif key == "review_play":
                    self.replay_playing = not self.replay_playing
                    self.replay_last_step = time.perf_counter()
                elif key == "review_start":
                    self.replay_playing = False
                    self.replay_index = 0
                    self.replay_board = self.replay_game.board()
                    self.replay_last_move = None
                    self.board_panel.animation_segments = []
                    self.board_panel.animation_source = None
                    self.board_panel.animation_capture_square = None
                    self.board_state = self.replay_board.copy()
                    self._sync_review_history()
                elif key == "review_end":
                    while self.replay_index < len(self.replay_moves):
                        self._replay_step(1)
                    self.replay_playing = False
                break

    def _set_difficulty_from_x(self, x):
        slider = self.menu_slider_rect
        if slider.w <= 0:
            return
        ratio = min(1.0, max(0.0, (x - slider.x) / slider.w))
        self.last_settings["difficulty"] = max(1, min(10,
            1 + round(ratio * 9)))

    def _draw_viewport_hint(self):
        if not hasattr(self, "viewport_font"):
            self.viewport_font = pygame.font.SysFont("consolas", 13)
        txt = "3D VIEWPORT   MMB-drag: orbit   wheel: zoom"
        self.screen.blit(self.viewport_font.render(txt, True, GREY), (14, 12))

    def _draw_board_panel(self, info, fly_move, full_screen=False):
        p = self.board_panel
        p.set_full_screen(full_screen)
        p.draw(self.screen, self.board_state,
               last_move=info.get("last_chess_move"),
               fly_move=fly_move,
               status=info.get("status"),
               game_over=info.get("game_over"),
               clocks=info.get("clocks"))
        if info.get("game_over"):
            fnt = self.board_panel.ui_font
            ts = fnt.render("[SPACE] new game vs fly   [ESC] quit", True, WHITE)
            self.screen.blit(ts, (self.board_panel.x0 + 2,
                                  self.board_panel.y0 + 8 * SQ_PX + 52))

    def _grid(self):
        cam = self.cam
        for i in range(-30, 31, 6):
            a, b = v3(i, -6, -30), v3(i, -6, 30)
            p1, _ = project(a, cam)
            p2, _ = project(b, cam)
            pygame.draw.line(self.screen, (35, 35, 45), p1, p2, 1)
            a, b = v3(-30, -6, i), v3(30, -6, i)
            p1, _ = project(a, cam)
            p2, _ = project(b, cam)
            pygame.draw.line(self.screen, (35, 35, 45), p1, p2, 1)


def create_agent(settings):
    gateway = ConnectomeGateway(NEUPRINT_TOKEN, NEUPRINT_SERVER,
                                NEUPRINT_DATASET,
                                offline=settings["offline"])
    agent = FlyChessAgent(gateway)
    weights = WEIGHTS_PATH
    if os.path.isfile(weights):
        loaded = agent.net.load_weights(weights)
        print("Loaded %d learned weight layers from %s" % (loaded, weights))
    return agent


def run_menu_session(viz, agent, settings):
    """Main-menu loop and GUI-backed launch of play/training sessions."""
    viz.last_settings = settings
    viz.menu_session_active = True
    agent_offline = bool(settings["offline"])
    while viz.enabled and viz.screen_state != "quit":
        if viz.screen_state in ("menu", "settings", "training", "replays", "review"):
            handle_input(viz)
            if (viz.screen_state == "menu"
                    and bool(viz.last_settings["offline"]) != agent_offline):
                agent = create_agent(viz.last_settings)
                agent.body = viz.body
                agent_offline = bool(viz.last_settings["offline"])
            if viz.screen_state in ("menu", "settings", "training", "replays", "review"):
                viz.render(agent.net.last_sensory, agent.net.last_motor,
                           info={"mode": "MENU", "eco": False})
                continue
        if viz.screen_state == "playing":
            action = viz.menu_action or "play"
            viz.menu_action = None
            settings = viz.last_settings
            save_settings(settings)
            if action == "training":
                games = settings["training_games"]
                selfplay, interactive, think = True, False, \
                    difficulty_think_seconds(settings["difficulty"])
                viz.training_mode = True
            else:
                games, selfplay, interactive = 1, False, True
                think = difficulty_think_seconds(settings["difficulty"])
                viz.training_mode = False
            viz.eco_only = settings["eco_mode"]
            viz.body.eco = settings["eco_mode"]
            agent.body = viz.body
            agent.realistic = settings["realistic_mode"]
            viz.board_panel.set_full_screen(settings["eco_mode"])
            viz.board_panel.reset()
            viz.paused = False
            rng = random.Random()
            run_match(
                agent, games, 160, 8.0, selfplay, viz, rng,
                lambda *_: None, interactive=interactive, think=think,
                realistic=settings["realistic_mode"],
                animation_seconds=settings["animation_seconds"],
                return_to_menu=(action != "training"))
            if viz.training_mode:
                try:
                    saved = agent.net.save_weights(WEIGHTS_PATH)
                    print("[training] saved learned weights -> %s" % saved)
                except OSError as exc:
                    print("[training] could not save weights: %s" % exc)
            viz.training_mode = False
            viz.menu_action = None
            viz.screen_state = "menu"
    viz.menu_session_active = False
    return settings


def main():
    ap = argparse.ArgumentParser(description="Fruit Fly Connectome Chess Bot")
    ap.add_argument("--menu", action="store_true",
                    help="open the interactive main menu (default when no mode flags are used)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--train", action="store_true",
                    help="headless self-play training mode")
    ap.add_argument("--weights", default=None,
                    help="load learned connectome weights from .npz")
    ap.add_argument("--save-weights", default=None,
                    help="save learned connectome weights to .npz")
    ap.add_argument("--no-menu", action="store_true",
                    help="run the legacy direct game flow instead of the new menu")
    ap.add_argument("--ply-limit", type=int, default=160)
    ap.add_argument("--speed", type=float, default=8.0,
                    help="plies per second (0 = as fast as possible)")
    ap.add_argument("--think", type=float, default=THINK_TIME,
                    help="seconds the fly deliberates per move "
                         "(0 = pure connectome reflex)")
    ap.add_argument("--realistic", action="store_true",
                    help="realistic fly hardware: the fly knows the rules "
                         "but decides by innate reflex + dopamine-learned "
                         "connectome drive only - no search, no thinking "
                         "pause, snap moves in ~0.25 s")
    ap.add_argument("--eco", action="store_true",
                    help="eco mode: legacy hard-coded idle animation - no "
                         "motor-neuron readout, no walking, no proprioception "
                         "(connectome budget goes to chess only)")
    ap.add_argument("--selfplay", action="store_true",
                    help="connectome plays both colours (autoplay)")
    ap.add_argument("--watch", action="store_true",
                    help="autopilot: fly (Black) vs random White, no human")
    ap.add_argument("--offline", action="store_true",
                    help="skip the neuPrint API and use the mock connectome")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    if args.offline:
        settings["offline"] = True
    legacy_requested = args.no_menu or any((args.watch, args.selfplay, args.train,
                            args.no_gui, args.eco, args.realistic,
                            args.weights is not None,
                            args.save_weights is not None, args.games != 1,
                            args.ply_limit != 160, args.speed != 8.0,
                            args.think != THINK_TIME))
    if not args.no_menu and not legacy_requested:
        if pygame is None:
            raise SystemExit("pygame is required for the interactive menu")
        if not pygame.font.get_init():
            pygame.font.init()
        viz = Visualizer(enabled=True)
        agent = create_agent(settings)
        agent.body = viz.body
        viz.screen_state = "menu"
        try:
            settings = run_menu_session(viz, agent, settings)
        finally:
            viz._cancel_review_analysis()
            if viz.review_analysis_stop is not None:
                viz.review_analysis_stop.set()
            thread = viz.review_analysis_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
            try:
                pygame.quit()
            except Exception:
                pass
        return
    settings["difficulty"] = max(1, min(10, int(round(args.think * 10 / 8)))) \
        if args.think != THINK_TIME else settings["difficulty"]
    if args.eco:
        settings["eco_mode"] = True
    if args.realistic:
        settings["realistic_mode"] = True

    # Interactive click-to-play is the default; training is headless
    # self-play so repeated games do not wait on a human or a window.
    if args.train:
        args.selfplay = True
        args.no_gui = True
    interactive = not (args.watch or args.selfplay or args.no_gui)

    rng = random.Random(args.seed)
    if np is not None and args.seed is not None:
        np.random.seed(args.seed)

    # ---------------- 1. neuPrint setup -------------------------------------
    print("=" * 72)
    print("FRUIT FLY CONNECTOME CHESS BOT")
    print("=" * 72)
    gateway = ConnectomeGateway(NEUPRINT_TOKEN, NEUPRINT_SERVER,
                                NEUPRINT_DATASET, offline=args.offline)

    # ---------------- 2+3. chess agent with dopamine loop --------------------
    agent = FlyChessAgent(gateway)
    if args.weights:
        loaded = agent.net.load_weights(args.weights)
        print("Loaded %d learned weight layers from %s" % (loaded, args.weights))
    print("Brain split: chess deliberation = search + connectome (expensive);"
          "\n             locomotion = innate VNC reflex (cheap, frozen while"
          "\n             thinking, no learning, no search).")
    print("Connectome: %d sensory (ME R) -> %s -> %d motor (VNC/DN) neurons"
          % (len(agent.net.sensory), " -> ".join(
              r for r in agent.net.regions[1:-1]), len(agent.net.motor)))

    # ---------------- 4. visualizer -----------------------------------------
    viz = Visualizer(enabled=not args.no_gui)
    agent.body = viz.body      # the agent senses and moves through its body
    viz.body.eco = args.eco
    viz.eco_only = args.eco
    viz.board_panel.set_full_screen(args.eco)
    # Explicit command-line game modes keep their legacy direct-game UI;
    # the graphical menu is only the default no-arguments launch.
    viz.screen_state = "playing"
    agent.realistic = args.realistic
    if args.realistic:
        print("REALISTIC mode: fly hardware.  No search, no deliberation - "
              "innate instincts + dopamine-learned connectome drive choose "
              "each move in ~%.2f s, like the real animal." % FLY_REACTION_TIME)
    if args.eco:
        print("ECO mode: legacy hard-coded fly animation.  No motor-neuron "
              "readout, no walking, no proprioception - the connectome's "
              "compute budget is reserved for chess.")
    if args.train:
        print("TRAIN mode: connectome self-play for %d game(s), %d plies/game"
              % (args.games, args.ply_limit))

    # ---------------- 5. engine loop ----------------------------------------
    status = lambda g, G, p, mv, sc, dr, dp, da: None
    final_score = 0.0
    try:
        final_score = run_match(agent, args.games, args.ply_limit, args.speed,
                                args.selfplay, viz, rng, status,
                                interactive=interactive, think=args.think,
                                realistic=args.realistic)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if viz.enabled:
            pygame.quit()
        if args.save_weights:
            print("Saved learned weights -> %s" %
                  agent.net.save_weights(args.save_weights))
    print("Final fly score: %.1f   DA reward spikes: %d   penalties: %d"
          % (final_score, agent.da_reward, agent.da_penalty))


if __name__ == "__main__":
    main()
