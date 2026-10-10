# GamingBaguette15 — Downloads & Projects

A small download shelf for projects and tools I use. Please read the notes and limitations before downloading or running anything.

> ⚠ Please read this README.md or you will probably not know how to use most stuff.

## CreamInstaller

Credit to FroggMaster for CreamInstaller. It does not download DLC files. My understanding is that it can work with some games such as Plague Inc. when the DLC content is already present in the game files. It will not work for games where DLC has to be downloaded separately. Compatibility varies by title and game version.

Only use tools and game content you are authorized to use, and follow the game's and platform's rules. The [CreamInstaller release](https://github.com/GamingBaguette15/GamingBaguette15.github.io/releases/tag/Stuff) is linked here for download. This page does not provide instructions to bypass purchases, access controls, or online checks.

## FruitFlyChessBot

A chess bot experiment inspired by a fruit-fly brain/connectome map. Download the 7z archive, extract it with [7-Zip](https://www.7-zip.org/), and run `main.bat` on Windows. The launcher checks for Python 3.9 or newer, installs the core dependencies, and starts the bot; **Python 3.12 is recommended**.

The bot can use mock/offline connectome data when neuPrint is unavailable. If you have authorized neuPrint API access, set `NEUPRINT_TOKEN` in your environment; never paste API tokens into source files or commit them.

- [Download FruitFlyChessBot.7z](Main%20Stuff/FruitFlyChessBot.7z)
- [Browse the source folder](Main%20Stuff/FruitFlyChessBot/)

## VR APK modding

The latest Java version is required, and you will need some coding knowledge to understand and edit most of the files. Use these tools only with APKs you own or are authorized to modify, and keep testing to offline, single-player use.

Games that require online authentication—such as Gorilla Tag—are not supported. This is not for bypassing authentication, anti-cheat, or multiplayer protections. Back up your files before experimenting; I can't guarantee that a modified APK will work on a particular headset or game version.

- [Download APK modding tools 7z](Main%20Stuff/APK%20modding.7z)
- [Browse the APK modding folder](Main%20Stuff/APK%20modding/)

## General notes

- Most projects need Python; Python 3.12 is preferred. The VR APK modding tools require the latest Java version.
- Extract `.7z` downloads with 7-Zip before use. Most projects are source files or tools, not a promise of one-click setup; read their included files before running anything.
- Scan downloads with your security software and only run files you trust.
- Compatibility and requirements can change. Check the individual project files for current details.