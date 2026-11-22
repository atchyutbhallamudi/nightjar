"""Drive Bot._act through situations the recorded match never shows (second weapons, swaps, offers,
supply drops, mortar shells, enemies behind buildings, retreating) and check every action against the
protocol schema with the kit's own validator. Any exception fails the test (the wrapper would hide it)."""
import copy
import gzip
import importlib.util
import json
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")

import mechfight as mf  # noqa: E402
from bot import Bot  # noqa: E402


def load_schema():
    spec = importlib.util.spec_from_file_location("kit_check", os.path.join(KIT, "check", "check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Schema(os.path.join(KIT, "mechfight.schema.json"))


@unittest.skipUnless(os.path.exists(STREAM), "starter kit not unpacked in ./kit")
class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(STREAM, "rt") as fh:
            msgs = [json.loads(line) for line in fh]
        cls.hello = msgs[0]
        cls.hello["map"]["dir"] = os.path.join(KIT, "arena")
        cls.base = [m for m in msgs if m.get("type") == "obs"][120]
        cls.schema = load_schema()
        cls.cfg = cls.hello["config"]
        mf.draw.caps = cls.cfg["debug"]
        cls.proto_bot = Bot(cls.hello)
        cls.world = cls.proto_bot.world

    # ---------- helpers ----------

    def fresh_bot(self):
        b = Bot.__new__(Bot)
        b.__dict__.update(copy.copy(self.proto_bot.__dict__))
        # reset per-match state that copy.copy shares
        for k in ("threats", "bad_pickups", "bad_goals", "_safe_cache"):
            setattr(b, k, {})
        b.path = None
        b.path_goal = None
        b.prev_t = 0.0
        return b

    def weapon(self, kind, **state):
        w = {"kind": kind, "cooldown": 0.0, "stats": dict(self.cfg["weapons"][kind])}
        if kind == "gun":
            w.update(ammo=12, reload=0)
        if kind == "laser":
            w.update(charge=0.0)
        w.update(state)
        return w

    def obs(self, me_pos=None, weapons=None, mechs=(), **extra):
        o = copy.deepcopy(self.base)
        o["events"], o["sounds"], o["beams"], o["projectiles"] = [], [], [], []
        o["mechs"], o["pickups"], o["cubes"], o["crates"], o["drops"] = list(mechs), [], [], [], []
        o["offer"] = None
        if me_pos is not None:
            o["self"]["pos"] = list(me_pos)
            o["self"]["vel"] = [0.0, 0.0, 0.0]
        o["self"]["hp"] = o["self"]["max_hp"]
        o["self"]["weapons"] = weapons or [self.weapon("gun"), None]
        o.update(extra)
        return o

    def enemy(self, pos, eid=7, **kw):
        e = {"id": eid, "kind": "mech", "pos": list(pos), "vel": [3.0, -2.0, 0.0], "aim": [0.0, 0.0],
             "hp": 1400, "max_hp": 1400, "cubes": 0, "weapons": ["gun", "laser"], "charging": False}
        e.update(kw)
        return e

    def run_ticks(self, bot, o, n=8):
        acts = []
        for k in range(n):
            oo = copy.deepcopy(o)
            oo["tick"] = o["tick"] + 2 * k
            oo["t"] = round(o["t"] + 0.1 * k, 2)
            mf.draw.clear()
            act = bot._act(oo)
            act["tick"] = oo["tick"]
            d = mf.draw.take()
            if d:
                act["debug"] = d
            json.dumps(act)
            errs = self.schema.errors(act, "Action")
            self.assertFalse(errs, f"schema errors {errs} in {act}")
            self.assertEqual(len(act["fire"]), 2)
            acts.append(act)
        return acts

    def open_pair(self, dist, blocked):
        """Two ground points `dist` apart, with (blocked=True) or without a building between eye lines."""
        w = self.world
        me = self.base["self"]["pos"]
        for r in range(0, 400, 20):
            for k in range(24):
                a = (me[0] + r * math.cos(k), me[1] + r * math.sin(k))
                for j in range(12):
                    yaw = j * math.pi / 6
                    b = (a[0] + dist * math.cos(yaw), a[1] + dist * math.sin(yaw))
                    za, zb = w.ground(*a), w.ground(*b)
                    if w.inside_solid(a[0], a[1], za + 2) or w.inside_solid(b[0], b[1], zb + 2):
                        continue
                    hit = w.building_between([a[0], a[1], za + 7], [b[0], b[1], zb + 4.5])
                    if hit == blocked and (blocked or not w.terrain_between([a[0], a[1], za + 7], [b[0], b[1], zb + 4.5])):
                        return [a[0], a[1], za], [b[0], b[1], zb]
        self.skipTest("no suitable point pair found")

    # ---------- scenarios ----------

    def test_laser_charge_states(self):
        a, b = self.open_pair(100, blocked=False)
        for charge, cooldown in ((0.0, 0.0), (0.3, 0.0), (0.75, 0.0), (0.0, 1.2)):
            bot = self.fresh_bot()
            weapons = [self.weapon("gun"), self.weapon("laser", charge=charge, cooldown=cooldown)]
            acts = self.run_ticks(bot, self.obs(a, weapons, [self.enemy(b)]))
            self.assertTrue(all("aim" in x for x in acts))

    def test_laser_released_when_retreating(self):
        a, b = self.open_pair(100, blocked=False)
        bot = self.fresh_bot()
        bot.ffa = True  # free-for-all rules: retreat early with two enemies on us
        weapons = [self.weapon("gun"), self.weapon("laser", charge=0.5)]
        o = self.obs(a, weapons, [self.enemy(b, vel=[0, 0, 0]), self.enemy([b[0] + 10, b[1], b[2]], eid=8)])
        o["self"]["hp"] = 300
        # torso already on the target: the partial charge should be fired (trigger let go) right away
        muzzle_z = a[2] + 7
        o["self"]["aim"] = [math.atan2(b[1] - a[1], b[0] - a[0]),
                            math.atan2(b[2] + 4.5 - muzzle_z, math.hypot(b[0] - a[0], b[1] - a[1]))]
        acts = self.run_ticks(bot, o, n=3)
        self.assertFalse(acts[-1]["fire"][1], "a laser charge should be let go while retreating")

    def test_lightning_close_and_mortar_behind_cover(self):
        a, b = self.open_pair(30, blocked=False)
        bot = self.fresh_bot()
        weapons = [self.weapon("gun"), self.weapon("lightning")]
        self.run_ticks(bot, self.obs(a, weapons, [self.enemy(b)]))
        a, b = self.open_pair(120, blocked=True)
        bot = self.fresh_bot()
        weapons = [self.weapon("gun"), self.weapon("mortar")]
        acts = self.run_ticks(bot, self.obs(a, weapons, [self.enemy(b)]), n=12)
        self.assertFalse(any(x["fire"][0] for x in acts), "gun must not fire into a building")

    def test_swap_when_both_slots_full(self):
        a, _ = self.open_pair(50, blocked=False)
        bot = self.fresh_bot()
        weapons = [self.weapon("gun"), self.weapon("mortar")]
        pickup = {"id": 8123, "kind": "laser", "pos": [a[0] + 1.0, a[1], a[2]]}
        o = self.obs(a, weapons)
        o["pickups"] = [pickup]
        act = self.run_ticks(bot, o, n=1)[0]
        self.assertEqual(act.get("take"), 8123)
        self.assertEqual(act.get("drop"), 1, "should drop the mortar, not the gun")

    def test_offer_supply_drop_and_cubes(self):
        a, _ = self.open_pair(50, blocked=False)
        bot = self.fresh_bot()
        o = self.obs(a)
        o["offer"] = {"id": 1, "expires": o["t"] + 5, "choices": [
            {"id": "gun_velocity", "name": "x", "desc": "x"}, {"id": "armor", "name": "y", "desc": "y"},
            {"id": "sensors", "name": "z", "desc": "z"}]}
        o["drops"] = [{"id": 20000, "pos": [a[0] + 40, a[1], a[2]], "radius": 8, "lands_in": 3.0, "capture_s": 4,
                       "progress": 0, "holder": None}]
        o["cubes"] = [{"id": 5001, "pos": [a[0] - 30, a[1], a[2]]}]
        acts = self.run_ticks(bot, o, n=3)
        self.assertEqual(acts[0].get("pick"), 1, "armor is preferred")

    def test_mortar_shell_dodge(self):
        a, _ = self.open_pair(50, blocked=False)
        bot = self.fresh_bot()
        o = self.obs(a)
        g = self.cfg["weapons"]["mortar"]["gravity"]
        # a shell 1 s from landing right on us, coming from the east
        shell = {"id": 900, "kind": "mortar", "pos": [a[0] + 30, a[1], a[2] + 11 + 0.5 * g],
                 "vel": [-30.0, 0.0, -11.0 + 0.0]}
        o["projectiles"] = [shell]
        act = self.run_ticks(bot, o, n=1)[0]
        mv = act["move"]
        self.assertGreater(math.hypot(mv[0], mv[1]), 0.9, f"should run from the splash, move={mv}")

    def test_target_behind_building_holds_fire(self):
        a, b = self.open_pair(90, blocked=True)
        bot = self.fresh_bot()
        acts = self.run_ticks(bot, self.obs(a, None, [self.enemy(b)]), n=10)
        self.assertFalse(any(x["fire"][0] for x in acts))


if __name__ == "__main__":
    unittest.main()
