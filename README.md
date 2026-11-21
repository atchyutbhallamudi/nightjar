# mech-royale

A bot for the **Mech Royale** track of the Builders Buildathon (Nov 20–22, 2026). It pilots one mech in a
16-mech battle royale. Score = mean of `16 − place` per match, so surviving matters more than kills.

## How the bot plays

Every 0.1 s of match time the bot (`bot.py`) picks one mode:

| Mode | When | What it does |
| --- | --- | --- |
| `zone` | outside the zone, or the next circle closes soon | runs (boosting if needed) into the next safe circle |
| `retreat` | HP low (< 38%, < 45% near a laser, < 60% vs two enemies) | finds a spot no enemy can see, waits there to regenerate (5 s with no damage, then 45 HP/s) |
| `fight` | an enemy mech is in view | holds a range that suits its weapons (laser 60–140 m, gun 40–100 m, lightning < 30 m), strafes, sidesteps visible laser charge-ups |
| `weapon` / `cube` / `drop` | no enemy in view | walks to better weapons (laser > gun > lightning > mortar), power cubes, supply drops; shoots crates and (early) drones for cubes |
| `roam` / `evade` | nothing else | walks quietly (slow walking makes no sound) inside the safe circle; sidesteps if shot from out of view |

Aiming leads moving targets for the gun, aims straight for the hitscan laser, and solves the ballistic arc for
the mortar. It only fires when the torso is actually on target and no building or hill blocks the shot (`world.py`
loads the map and does the line-of-sight math). Upgrades prefer armor, HP and regen, then damage.

## Files

| Path | What it is |
| --- | --- |
| `bot.py` | The bot (entry point). |
| `world.py` | Map loading: terrain height, buildings, line-of-sight and wall checks. |
| `mechfight.py` | The organizers' protocol wrapper from the starter kit, unchanged. |
| `Dockerfile` | The image the scorer builds. Standard library only; no secrets or environment variables are needed. |
| `tests/` | Unit tests: map checks against a recorded match, and aiming math. |
| `tools/replay.py` | Feeds the recorded match through the bot and prints its decisions (shows crashes the wrapper would hide). |

## Run and test locally

You need Python 3.9+ (3.13 recommended). Docker is optional locally; the scorer builds the image itself.

1. Download the starter kit (the bot's tests use its recorded match and map) and unpack it into `kit/`:

   ```powershell
   curl.exe -L -o mechfight-kit.tar.gz https://www.joinbuilders.org/buildathon/tracks/builders/files/mechfight-kit.tar.gz
   mkdir kit; tar -xzf mechfight-kit.tar.gz -C kit --strip-components=1
   ```

2. Run the tests, the replay, and the organizers' conformance check:

   ```powershell
   python -m unittest discover -s tests
   python tools/replay.py
   python kit/check/check.py -- python -u bot.py
   ```

3. Optional, with Docker Desktop installed: check the image exactly the way a match runs it:

   ```powershell
   docker build -t my-bot .
   python kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${PWD}/kit/arena:/arena:ro" my-bot
   ```

## Submit

1. Push this repository to a **public** GitHub repo, with `Dockerfile` at the top level (max 20 MB).
2. Sign in at https://www.joinbuilders.org/buildathon, open the Mech Royale track page, and submit the repo URL.
3. Each submission is built, checked, and scored over 48 practice matches (4 submissions per hour per team).
4. Keep the repo public, with every submitted commit, until results are published.

## Credits and license

`mechfight.py` comes from the organizers' `mechfight-kit`. Everything else is released under the MIT License
(see `LICENSE`), as the Buildathon rules require.
