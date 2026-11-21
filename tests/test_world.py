"""Checks world.py against the kit's recorded match (kit/check/stream.jsonl.gz).

Run from the repo root:  python -m unittest discover -s tests
Needs the starter kit unpacked in ./kit (see README).
"""
import gzip
import json
import math
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")

from world import World  # noqa: E402


@unittest.skipUnless(os.path.exists(STREAM), "starter kit not unpacked in ./kit")
class WorldAgainstRecording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t0 = time.perf_counter()
        cls.world = World(os.path.join(KIT, "arena"), ["v2/world.json", "v2/terrain.i16"])
        cls.load_s = time.perf_counter() - t0
        with gzip.open(STREAM, "rt") as fh:
            msgs = [json.loads(line) for line in fh]
        cls.obs = [m for m in msgs if m.get("type") == "obs"]

    def test_loads_fast(self):
        self.assertLess(self.load_s, 3.0, f"map load took {self.load_s:.2f}s")

    def test_grounded_height_matches_terrain_or_roof(self):
        errs = []
        for o in self.obs:
            me = o["self"]
            if not me["grounded"]:
                continue
            x, y, z = me["pos"]
            g = self.world.ground(x, y)
            on_roof = any(abs(p.top - z) < 0.5 and p.contains_xy(x, y) for p in self.world.prisms)
            if not on_roof:
                errs.append(abs(g - z))
        self.assertTrue(errs, "no grounded frames")
        errs.sort()
        median = errs[len(errs) // 2]
        self.assertLess(median, 0.6, f"median terrain error {median:.2f} m")

    def test_newly_seen_mechs_are_not_behind_buildings(self):
        seen_prev = set()
        checked = blocked = 0
        for o in self.obs:
            me = o["self"]
            eye = [me["pos"][0], me["pos"][1], me["pos"][2] + 8]
            now = set()
            for m in o["mechs"]:
                now.add(m["id"])
                if m["id"] in seen_prev:
                    continue
                p = m["pos"]
                h = 9 if m["kind"] == "mech" else 3.5
                center = [p[0], p[1], p[2] + h / 2]
                top = [p[0], p[1], p[2] + h]
                checked += 1
                if self.world.building_between(eye, center) and self.world.building_between(eye, top):
                    blocked += 1
            seen_prev = now
        self.assertGreater(checked, 0)
        self.assertLessEqual(blocked, max(1, checked // 10), f"{blocked}/{checked} new sightings looked blocked")

    def test_damage_bearing_convention(self):
        diffs = []
        for o in self.obs:
            me = o["self"]
            by_id = {m["id"]: m for m in o["mechs"]}
            for e in o["events"]:
                if e["type"] == "damaged" and e.get("by") in by_id and e.get("bearing") is not None:
                    src = by_id[e["by"]]["pos"]
                    yaw = math.atan2(src[1] - me["pos"][1], src[0] - me["pos"][0])
                    d = (e["bearing"] - yaw + math.pi) % (2 * math.pi) - math.pi
                    diffs.append(abs(d))
        self.assertTrue(diffs)
        diffs.sort()
        self.assertLess(diffs[len(diffs) // 2], 0.3)

    def test_line_of_sight_speed(self):
        o = self.obs[len(self.obs) // 2]
        me = o["self"]["pos"]
        a = [me[0], me[1], me[2] + 7]
        t0 = time.perf_counter()
        n = 0
        for k in range(40):
            yaw = k * math.pi / 20
            b = [me[0] + 180 * math.cos(yaw), me[1] + 180 * math.sin(yaw), me[2] + 4]
            self.world.clear_line(a, b)
            n += 1
        per = (time.perf_counter() - t0) / n * 1000
        self.assertLess(per, 2.0, f"clear_line took {per:.2f} ms")


if __name__ == "__main__":
    unittest.main()
