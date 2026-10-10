# GamingBaguette15 — Downloads & Projects

A small download shelf for projects and tools I use. Read each item's notes and limitations before downloading. Files and links may change as the page grows.

## CreamInstaller

CreamInstaller is a third-party DLC-unlocker utility. It does **not** download DLC files. It only applies to supported games when the DLC files are already installed locally; it will not work when those files are not present or need to be downloaded separately. I understand it may work with titles such as Plague Inc., but compatibility varies by game and may change over time.

Use it only with games and content you are authorized to access, and follow the game's and platform's rules. I can't help bypass purchases, access controls, or online checks. This is a personal tool listing, not an endorsement or guarantee of compatibility. No download is hosted or linked here.

## FruitFlyChessBot

A Python chess-bot experiment inspired by a fruit-fly brain/connectome map. The folder includes the source, settings, learned-weight data, and Windows launch scripts. **Python 3.12 is recommended.** The launcher requires Python 3.9 or newer and installs the core dependencies if needed.

From the project folder, install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Then run `main.bat` on Windows, or start the script with `python fly_connectome_chess.py` on another platform. The project uses a mock/offline connectome when neuPrint is unavailable. If you have authorized neuPrint API access, set `NEUPRINT_TOKEN` in your environment; do not put API tokens in source files or commit them.

- [Download the Python script](Main%20Stuff/FruitFlyChessBot/fly_connectome_chess.py)
- [View requirements](Main%20Stuff/FruitFlyChessBot/requirements.txt)
- [Browse project files](Main%20Stuff/FruitFlyChessBot/)

## VR APK modding

**Warning: Please read the README.md or you will probably not know how to use most stuff.** The latest Java version is required to use the APK modding tools. These are tools and files I use for modding VR APKs. You will need some coding knowledge to understand and edit the files. Use them only with APKs you own or are authorized to modify, and keep testing to offline, single-player use.

Games that require online authentication—such as Gorilla Tag—are not supported. This is not for bypassing authentication, anti-cheat, or multiplayer protections. Back up your files before experimenting; I can't guarantee that a modified APK will work on a particular headset or game version.

- [Browse the APK modding folder](MainStuff/APK modding/)

## General notes

- Most projects need Python; Python 3.12 is preferred. Check each project's requirements before getting started. The VR APK modding tools require the latest Java version.
- Most projects are source files or tools, not a promise of one-click setup. Read their included files before running anything.
- Scan downloads with your security software and only run files you trust.
- Compatibility and requirements can change. If something is unclear, check back for updated notes.
