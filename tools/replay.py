"""Feed the kit's recorded match straight into Bot.act() and print what it decides.

Unlike check/check.py this raises on any exception (the wrapper would swallow it) and shows
the bot's mode, movement, aim and fire choices. The recording does not react to the bot.

    python tools/replay.py            # summary every 2 seconds of match time
    python tools/replay.py --every 1  # every observation
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mechfight as mf  # noqa: E402
from bot import Bot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"))
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()
    with gzip.open(args.stream, "rt") as fh:
        msgs = [json.loads(line) for line in fh]
    hello = msgs[0]
    hello["map"]["dir"] = os.path.join(ROOT, "kit", "arena")
    mf.draw.caps = hello["config"]["debug"]
    bot = Bot(hello)
    modes: Counter = Counter()
    fires = Counter()
    worst = 0.0
    n = 0
    for m in msgs:
        if m.get("type") != "obs":
            continue
        mf.draw.clear()
        t0 = time.perf_counter()
        act = bot._act(m)
        worst = max(worst, (time.perf_counter() - t0) * 1000)
        json.dumps(act)  # must serialize
        label = (mf.draw.take() or {}).get("label", "")
        mode = label.split(" ")[0]
        modes[mode] += 1
        for i, f in enumerate(act.get("fire", [])):
            if f:
                fires[i] += 1
        if n % args.every == 0:
            me = m["self"]
            vis = [(e["id"], round(((e["pos"][0] - me["pos"][0]) ** 2 + (e["pos"][1] - me["pos"][1]) ** 2) ** 0.5))
                   for e in m["mechs"] if e["kind"] == "mech"]
            print(f"t={m['t']:5.1f} {label:32s} move={act.get('move')} boost={act.get('boost')} "
                  f"aim={act.get('aim')} myaim={[round(a, 2) for a in me['aim']]} fire={act.get('fire')} "
                  f"enemies={vis} take={act.get('take')}")
        n += 1
    print(f"\n{n} decisions, worst {worst:.1f} ms; modes {dict(modes)}; fire counts per slot {dict(fires)}")


if __name__ == "__main__":
    main()
