"""Movement-only harness: drive the real Bot around the real map with simple physics, no opponents.

It answers "does the bot get where it is going?" without guessing how enemies fight:
  * travel  - random start points walk to a far zone centre
  * loot    - random start points walk to a weapon pickup and take it
  * zone    - a full match zone schedule; counts ticks spent outside the moving circle

Physics follow protocol.md section 7 closely enough for navigation: acceleration toward the commanded
velocity, wall sliding, 1.2 m step-ups, jumping, jetting with energy, landing on roofs.

    python tools/navsim.py --runs 60
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mechfight as mf  # noqa: E402
from bot import Bot  # noqa: E402
from world import World  # noqa: E402

DT = 0.05  # one sim tick; the bot decides every 2 ticks
TIMES = []  # milliseconds per bot decision


def decide(bot, obs):
    t0 = time.perf_counter()
    act = bot._act(obs)
    TIMES.append((time.perf_counter() - t0) * 1000)
    return act


class Body:
    def __init__(self, world, cfg, x, y):
        self.w = world
        m = cfg["mech"]
        self.m = m
        self.r, self.h, self.step = m["radius"], m["height"], m["step_up"]
        self.x, self.y = x, y
        self.z = self.floor(x, y, 1e9)
        self.vx = self.vy = self.vz = 0.0
        self.energy = m["energy"]
        self.spent_t = -9.0
        self.grounded = True
        self.aim = [0.0, 0.0]

    def _prisms_near(self, x, y):
        w = self.w
        i0, j0 = w._cell(x - self.r, y - self.r)
        i1, j1 = w._cell(x + self.r, y + self.r)
        out = set()
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                out.update(w.grid[j * w.gnx + i])
        return [w.prisms[k] for k in out]

    def floor(self, x, y, z):
        """Highest surface under the centre that the feet can stand on from height z."""
        f = self.w.ground(x, y)
        for p in self._prisms_near(x, y):
            if p.top <= z + self.step + 0.01 and p.top > f and p.contains_xy(x, y):
                f = p.top
        return f

    def blocked(self, x, y, z):
        if x < self.w.minx + self.r or x > self.w.maxx - self.r or y < self.w.miny + self.r or y > self.w.maxy - self.r:
            return True
        pts = [(x, y)] + [(x + self.r * math.cos(a), y + self.r * math.sin(a)) for a in [k * math.pi / 4 for k in range(8)]]
        for p in self._prisms_near(x, y):
            if p.base < z + self.h and p.top > z + self.step:
                for px, py in pts:
                    if p.contains_xy(px, py):
                        return True
        # ground too steep to walk up
        g = self.w.ground(x, y)
        return g > z + self.step + 0.3 and self.grounded

    def apply(self, act, t):
        m = self.m
        mv = act.get("move", [0, 0])
        mag = min(1.0, math.hypot(mv[0], mv[1]))
        dirx, diry = (mv[0] / mag, mv[1] / mag) if mag > 1e-6 else (0.0, 0.0)
        boost = act.get("boost", False)
        for _ in range(2):
            speed = m["walk_speed"]
            if self.grounded and boost and mag > 0 and self.energy > 0:
                speed = m["boost_speed"]
                self.energy -= m["boost_drain"] * DT
                self.spent_t = t
            want_x, want_y = dirx * mag * speed, diry * mag * speed
            acc = (m["ground_accel"] if self.grounded else m["air_accel"]) * DT
            dvx, dvy = want_x - self.vx, want_y - self.vy
            dv = math.hypot(dvx, dvy)
            if dv > acc:
                dvx, dvy = dvx / dv * acc, dvy / dv * acc
            self.vx += dvx
            self.vy += dvy
            if act.get("jump") and self.grounded:
                self.vz = m["jump_speed"]
                self.grounded = False
            if not self.grounded and boost and self.energy > 0:
                self.vz = min(self.vz + m["jet_accel"] * DT, m["max_rise_speed"])
                self.energy -= m["jet_drain"] * DT
                self.spent_t = t
            self.vz -= m["gravity"] * DT
            nx, ny = self.x + self.vx * DT, self.y + self.vy * DT
            if not self.blocked(nx, ny, self.z):
                self.x, self.y = nx, ny
            elif not self.blocked(nx, self.y, self.z):
                self.x, self.vy = nx, 0.0
            elif not self.blocked(self.x, ny, self.z):
                self.y, self.vx = ny, 0.0
            else:
                self.vx = self.vy = 0.0
            nz = self.z + self.vz * DT
            f = self.floor(self.x, self.y, max(self.z, nz))
            if nz <= f:
                self.z, self.vz, self.grounded = f, 0.0, True
            else:
                self.z, self.grounded = nz, False
                # slide off: if nothing under us within step, we are airborne
            if t - self.spent_t > m["energy_regen_delay"]:
                self.energy = min(m["energy"], self.energy + m["energy_regen"] * DT)
        # torso slews toward the commanded aim
        want = act.get("aim", self.aim)
        dy = (want[0] - self.aim[0] + math.pi) % (2 * math.pi) - math.pi
        lim = m["turn_rate"] * 0.1
        self.aim[0] += max(-lim, min(lim, dy))
        self.aim[1] += max(-m["pitch_rate"] * 0.1, min(m["pitch_rate"] * 0.1, want[1] - self.aim[1]))


def zone_at(cfg, t):
    """Current circle and the next one, following the config phases (protocol.md section 9)."""
    zc = cfg["zone"]
    c, r = list(zc["start_center"]), zc["start_radius"]
    t0 = 0.0
    centers = zone_at.centers
    for i, ph in enumerate(zc["phases"]):
        nc, nr = centers[i], ph["radius"]
        ss, se = t0 + ph["hold_s"], t0 + ph["hold_s"] + ph["shrink_s"]
        if t < se:
            if t <= ss:
                cur_c, cur_r = c, r
            else:
                f = (t - ss) / (se - ss)
                cur_c = [c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f]
                cur_r = r + (nr - r) * f
            return {"center": cur_c, "radius": cur_r, "next_center": nc, "next_radius": nr,
                    "shrink_start": ss, "shrink_end": se, "dps": ph["dps"]}, (cur_c, cur_r)
        c, r, t0 = nc, nr, se
    return {"center": c, "radius": 0.0, "next_center": c, "next_radius": 0.0,
            "shrink_start": t0, "shrink_end": t0, "dps": 260}, (c, 0.0)


def make_centers(cfg, rng):
    zc = cfg["zone"]
    c, r = zc["start_center"], zc["start_radius"]
    out = []
    for ph in zc["phases"]:
        nr = ph["radius"]
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0, max(0.0, r - nr))
        c = [c[0] + d * math.cos(a), c[1] + d * math.sin(a)]
        out.append(c)
        r = nr
    return out


def base_obs(body, cfg, t, tick, zone, in_zone, pickups=()):
    m = cfg["mech"]
    gun = dict(cfg["weapons"]["gun"])
    return {
        "type": "obs", "tick": tick, "t": round(t, 2), "alive": 16,
        "self": {"id": 0, "pos": [body.x, body.y, body.z], "vel": [body.vx, body.vy, body.vz],
                 "aim": list(body.aim), "hp": 1400, "max_hp": 1400, "energy": body.energy, "max_energy": m["energy"],
                 "grounded": body.grounded, "cubes": 0, "upgrades": [], "in_zone": in_zone,
                 "weapons": [{"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0, "stats": gun}, None],
                 "stats": dict(m)},
        "mechs": [], "projectiles": [], "crates": [], "cubes": [], "pickups": list(pickups), "sounds": [],
        "beams": [], "events": [], "drops": [], "zone": zone, "offer": None,
    }


def random_open_point(world, body_cls, cfg, rng, center=None, max_d=None):
    for _ in range(500):
        if center is None:
            x = rng.uniform(world.minx + 40, world.maxx - 40)
            y = rng.uniform(world.miny + 40, world.maxy - 40)
        else:
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(20, max_d)
            x, y = center[0] + d * math.cos(a), center[1] + d * math.sin(a)
            if not (world.minx + 20 < x < world.maxx - 20 and world.miny + 20 < y < world.maxy - 20):
                continue
        b = body_cls(world, cfg, x, y)
        if abs(b.z - world.ground(x, y)) < 0.5 and not b.blocked(x, y, b.z):
            return x, y
    raise RuntimeError("no open point")


def new_bot(hello):
    mf.draw.caps = hello["config"]["debug"]
    return Bot(hello)


def run_travel(world, hello, rng, runs):
    cfg = hello["config"]
    ok, ratios, fails = 0, [], []
    for k in range(runs):
        sx, sy = random_open_point(world, Body, cfg, rng)
        gx, gy = random_open_point(world, Body, cfg, rng)
        if math.hypot(gx - sx, gy - sy) < 150:
            continue
        body = Body(world, cfg, sx, sy)
        bot = new_bot(hello)
        straight = math.hypot(gx - sx, gy - sy) / cfg["mech"]["walk_speed"]
        limit = straight * 3 + 20
        zone = {"center": [gx, gy], "radius": 2000, "next_center": [gx, gy], "next_radius": 30,
                "shrink_start": 9999, "shrink_end": 9999, "dps": 0}
        t, tick, done = 0.0, 0, False
        while t < limit:
            act = decide(bot, base_obs(body, cfg, t, tick, zone, True))
            body.apply(act, t)
            t += 0.1
            tick += 2
            if math.hypot(body.x - gx, body.y - gy) < 35:
                done = True
                break
        if done:
            ok += 1
            ratios.append(t / straight)
        else:
            fails.append(((round(sx), round(sy)), (round(gx), round(gy)), (round(body.x), round(body.y))))
    n = ok + len(fails)
    ratios.sort()
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    print(f"travel: {ok}/{n} arrived; time vs straight line median {med:.2f}x, worst {max(ratios or [0]):.2f}x")
    for f in fails[:8]:
        print("   stuck: start", f[0], "goal", f[1], "ended at", f[2])
    return ok, n


def run_loot(world, hello, rng, runs):
    cfg = hello["config"]
    ok, fails = 0, []
    for k in range(runs):
        sx, sy = random_open_point(world, Body, cfg, rng)
        px, py = random_open_point(world, Body, cfg, rng, center=(sx, sy), max_d=140)
        body = Body(world, cfg, sx, sy)
        bot = new_bot(hello)
        pz = world.ground(px, py)
        pickup = {"id": 8000 + k, "kind": "laser", "pos": [px, py, pz]}
        zone = {"center": [sx, sy], "radius": 2000, "next_center": [sx, sy], "next_radius": 1500,
                "shrink_start": 9999, "shrink_end": 9999, "dps": 0}
        t, tick, took = 0.0, 0, False
        limit = math.hypot(px - sx, py - sy) / cfg["mech"]["walk_speed"] * 3 + 15
        while t < limit:
            act = decide(bot, base_obs(body, cfg, t, tick, zone, True, [pickup]))
            body.apply(act, t)
            if act.get("take") == pickup["id"] and math.hypot(body.x - px, body.y - py) <= cfg["loot"]["pickup_radius"]:
                took = True
                break
            t += 0.1
            tick += 2
        if took:
            ok += 1
        else:
            fails.append(((round(sx), round(sy)), (round(px), round(py)), (round(body.x), round(body.y))))
    print(f"loot: {ok}/{runs} picked up the laser")
    for f in fails[:8]:
        print("   missed: start", f[0], "pickup", f[1], "ended at", f[2])
    return ok, runs


def run_zone(world, hello, rng, runs):
    cfg = hello["config"]
    total_out, worst, deaths = 0, 0, 0
    for k in range(runs):
        zone_at.centers = make_centers(cfg, rng)
        sx, sy = random_open_point(world, Body, cfg, rng, center=cfg["zone"]["start_center"], max_d=500)
        body = Body(world, cfg, sx, sy)
        bot = new_bot(hello)
        t, tick, out, hp = 0.0, 0, 0, 1400.0
        end_t = sum(p["hold_s"] + p["shrink_s"] for p in cfg["zone"]["phases"][:-1]) + 5
        while t < end_t:
            zone, (cc, cr) = zone_at(cfg, t)
            inside = math.hypot(body.x - cc[0], body.y - cc[1]) <= cr
            act = decide(bot, base_obs(body, cfg, t, tick, zone, inside))
            body.apply(act, t)
            if not inside:
                out += 1
                hp -= zone["dps"] * 0.1
            t += 0.1
            tick += 2
        total_out += out
        worst = max(worst, out)
        deaths += hp <= 0
    print(f"zone: {runs} matches to t={end_t:.0f}s; outside the circle {total_out / runs / 10:.1f}s on average, "
          f"worst {worst / 10:.1f}s; would have died to the zone in {deaths}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", choices=["travel", "loot", "zone"])
    ap.add_argument("--zone-runs", type=int, default=0, help="zone matches (default: runs / 5)")
    args = ap.parse_args()
    with gzip.open(os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"), "rt") as fh:
        hello = json.loads(fh.readline())
    hello["map"]["dir"] = os.path.join(ROOT, "kit", "arena")
    world = World(hello["map"]["dir"], hello["map"]["files"])
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    if args.only in (None, "travel"):
        run_travel(world, hello, rng, args.runs)
    if args.only in (None, "loot"):
        run_loot(world, hello, rng, args.runs)
    if args.only in (None, "zone"):
        run_zone(world, hello, rng, args.zone_runs or max(4, args.runs // 5))
    if TIMES:
        ts = sorted(TIMES)
        print(f"decision time ms: p50 {ts[len(ts) // 2]:.1f}  p99 {ts[int(len(ts) * 0.99)]:.1f}  max {ts[-1]:.1f}  ({len(ts)} decisions)")
    print(f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
