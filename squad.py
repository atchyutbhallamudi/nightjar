"""Hunt-mode decision core, used while every death so far has been our own kill.

One pass per decision, in this order:
  observe   keep a track per mech (where, how hurt, armed with what, when it last hit us, can we shoot it)
  engage    decide which tracks are coming for us, pick one target and stay on it
  posture   press on, or step back for a moment when clearly losing
  move      zone first; then cover, fight position, a cube from a kill, loot, or the walk to the next circle
  shoot     one torso, so one aim: gun lead by default, the laser's straight line once its charge is worth it

Geometry (terrain, buildings, sight lines, routes) comes from world.py / nav.py through the owning Bot.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

TAU = 2 * math.pi
INF = float("inf")

UPGRADE_ORDER = [
    "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_damage", "gun_rate", "lightning_efficiency",
    "lightning_range", "servos", "gun_mag", "gun_velocity", "gyros", "lightning_chain", "reactor", "capacitor",
    "mortar_reload", "mortar_splash", "mortar_velocity", "sensors", "jets", "laser_pierce",
]
WEAPON_RANK = {"laser": 4, "lightning": 3, "mortar": 1}


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Squad:
    def __init__(self, bot: Any) -> None:
        self.b = bot
        self.cfg = bot.cfg
        self.rng = bot.rng
        self.tr: Dict[Any, Dict[str, Any]] = {}   # mechs
        self.np: Dict[Any, Dict[str, Any]] = {}   # drones
        self.strafe = 1.0
        self.strafe_until = 0.0
        self.retreat = False
        self.lose_at = -99.0
        self.last_threat = -99.0
        self.last_hurt = -99.0
        self.hurt_bearing = 0.0
        self.npc_hurt = -99.0
        self.npc_bearing = 0.0
        self.stick_id: Any = None
        self.stick_at = -99.0
        self.loot_id: Any = None
        self.loot_since = 0.0
        self.bad: Dict[Any, float] = {}
        self.hold: Optional[Tuple[float, float, float]] = None
        self.hold_start: Optional[float] = None
        self.cover_goal: Optional[List[float]] = None
        self.cover_at = -99.0
        self.cover_ok = False
        self.lat = 0.075            # seconds between an observation and our answer taking effect
        self.engage_dist = 60.0
        self.max_target = 240.0
        self.range_gun, self.range_laser, self.range_light = 85.0, 120.0, 28.0
        self.loot_radius, self.laser_seek = 150.0, 240.0

    # ------------------------------------------------------------------ small helpers

    def _hgt(self, tk: Dict[str, Any]) -> float:
        return self.b._size(tk)[1]

    def _center(self, tk: Dict[str, Any]) -> List[float]:
        p = tk["pos"]
        return [p[0], p[1], p[2] + self._hgt(tk) * 0.5]

    def _eye(self, tk: Dict[str, Any]) -> List[float]:
        p = tk["pos"]
        return [p[0], p[1], p[2] + self._hgt(tk) * 8.0 / 9.0]

    def _weapon(self, me: Dict[str, Any], kind: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        for i, w in enumerate(me["weapons"]):
            if w and w["kind"] == kind:
                return i, w
        return -1, None

    # ------------------------------------------------------------------ zone

    def _zone_track(self, z: Dict[str, Any], now: float) -> None:
        live = (z["center"][0], z["center"][1], float(z["radius"]))
        if self.hold_start != z["shrink_start"] or now <= float(z["shrink_start"]) or self.hold is None:
            self.hold_start = z["shrink_start"]
            self.hold = live

    def _zone_at(self, z: Dict[str, Any], tt: float) -> Tuple[float, float, float]:
        ss, se = float(z["shrink_start"]), float(z["shrink_end"])
        h = self.hold or (z["center"][0], z["center"][1], float(z["radius"]))
        if tt <= ss or se <= ss:
            return h
        f = clamp((tt - ss) / (se - ss), 0.0, 1.0)
        nc, nr = z["next_center"], float(z["next_radius"])
        return (h[0] + (nc[0] - h[0]) * f, h[1] + (nc[1] - h[1]) * f, h[2] + (nr - h[2]) * f)

    def _zone_now(self, z: Dict[str, Any], now: float) -> Tuple[float, float, float]:
        live = (z["center"][0], z["center"][1], float(z["radius"]))
        h = self.hold or live
        if now <= float(z["shrink_start"]) or abs(live[2] - h[2]) > 0.01 or abs(live[0] - h[0]) > 0.01:
            return live
        return self._zone_at(z, now)

    def _zone_deadline(self, z: Dict[str, Any], x: float, y: float, pad: float) -> float:
        """When the shrinking edge reaches (x, y); never, if the spot is inside the next circle."""
        nc, nr = z["next_center"], float(z["next_radius"])
        if math.hypot(x - nc[0], y - nc[1]) <= nr - pad:
            return INF
        ss, se = float(z["shrink_start"]), float(z["shrink_end"])
        for k in range(25):
            tt = ss + (se - ss) * k / 24
            c = self._zone_at(z, tt)
            if math.hypot(x - c[0], y - c[1]) > c[2] - pad:
                return tt
        return se

    def _zone_penalty(self, z: Dict[str, Any], x: float, y: float, now: float) -> float:
        c = self._zone_now(z, now)
        p = 0.0
        d = math.hypot(x - c[0], y - c[1]) - c[2]
        if d > -5:
            p += 150 + 5 * (d + 5)
        nc, nr = z["next_center"], float(z["next_radius"])
        dn = math.hypot(x - nc[0], y - nc[1]) - nr
        if dn > -5:
            soon = clamp(1 - (float(z["shrink_start"]) - now) / 25, 0.1, 1.0)
            p += soon * (25 + 0.8 * (dn + 5))
        return p

    def _zone_goal(self, z: Dict[str, Any], x: float, y: float) -> Tuple[float, float]:
        nc, nr = z["next_center"], float(z["next_radius"])
        if nr <= 30:
            return nc[0], nc[1]
        dx, dy = x - nc[0], y - nc[1]
        d = math.hypot(dx, dy)
        r = max(0.0, nr - clamp(0.3 * nr, 3.0, 30.0) - 5)
        if d < 1 or d <= r:
            return x, y
        return nc[0] + dx / d * r, nc[1] + dy / d * r

    # ------------------------------------------------------------------ observe

    def _observe(self, obs: Dict[str, Any], me: Dict[str, Any], now: float) -> None:
        w = self.b.world
        st = me.get("stats", {})
        x, y, zf = me["pos"]
        muzzle = [x, y, zf + float(st.get("muzzle_height", 7))]
        for m in obs["mechs"]:
            if m["id"] == me["id"]:
                continue
            tab = self.np if m["kind"] == "npc" else self.tr
            tk = tab.get(m["id"])
            if tk is None:
                tk = tab[m["id"]] = {"hit": -99.0}
            tk.update(id=m["id"], kind=m["kind"], pos=m["pos"], vel=m["vel"], aim=m["aim"], hp=float(m.get("hp", 1400)),
                      max_hp=float(m.get("max_hp", 1400) or 1400), cubes=m.get("cubes", 0),
                      weapons=m.get("weapons") or [], charging=bool(m.get("charging")), seen=now)
            dx, dy = m["pos"][0] - x, m["pos"][1] - y
            d = math.hypot(dx, dy)
            tk["dist"] = d
            los = w is None or w.clear_line(muzzle, self._center(tk))
            if not los and m["kind"] == "mech":
                p = m["pos"]
                los = w.clear_line(muzzle, [p[0], p[1], p[2] + self._hgt(tk) * 0.85])
            tk["los"] = los
            tk["closing"] = -((m["vel"][0] - me["vel"][0]) * dx + (m["vel"][1] - me["vel"][1]) * dy) / d if d > 1 else 0.0
        self.tr = {k: v for k, v in self.tr.items() if now - v["seen"] <= 20}
        self.np = {k: v for k, v in self.np.items() if now - v["seen"] <= 120}
        for e in obs.get("events", []):
            if e.get("type") == "damaged":
                if e.get("weapon") == "npc" and e.get("bearing") is not None:
                    self.npc_hurt, self.npc_bearing = now, float(e["bearing"])
                by = e.get("by")
                if by is not None:
                    self.last_hurt = now
                    if e.get("bearing") is not None:
                        self.hurt_bearing = float(e["bearing"])
                    if by in self.tr:
                        self.tr[by]["hit"] = now
                    if by in self.np:
                        self.np[by]["hit"] = now
            elif e.get("type") == "kill":
                self.tr.pop(e.get("target"), None)
        self._zone_track(obs["zone"], now)

    # ------------------------------------------------------------------ movement pieces

    def _travel(self, me: Dict[str, Any], gx: float, gy: float, now: float, zone: bool = False) -> Tuple[Optional[float], float]:
        if math.hypot(gx - me["pos"][0], gy - me["pos"][1]) < 0.8:
            return None, 0.0
        return self.b._nav_yaw(me, [gx, gy], now, zone=zone), 1.0

    def _pref_range(self, me: Dict[str, Any], tk: Dict[str, Any], override: Optional[float]) -> float:
        if override is not None:
            return override
        if tk.get("kind") == "npc":
            return 58.0
        _, light = self._weapon(me, "lightning")
        _, laser = self._weapon(me, "laser")
        r = self.range_gun
        if light is not None and me["energy"] > 12:
            r = self.range_light
        elif laser is not None:
            r = self.range_laser
        if light is None:
            if "lightning" in tk.get("weapons", []):
                r = max(r, 64.0)
            if "mortar" in tk.get("weapons", []):
                r = max(r, 80.0)
        return r

    def _fight_move(self, obs: Dict[str, Any], me: Dict[str, Any], tg: Dict[str, Any], seen: List[Dict[str, Any]],
                    want_los: float, now: float, override: Optional[float] = None) -> Tuple[Optional[float], float]:
        """Step toward the best of sixteen spots 6 m away (or stay): right distance from the target, a line of fire
        when we can shoot, out of every other enemy's sight, in the zone, near a dropped cube, along the strafe."""
        w = self.b.world
        x, y, zf = me["pos"]
        st = me.get("stats", {})
        mh = float(st.get("muzzle_height", 7))
        body_h = float(st.get("height", 9)) * 0.6
        rp = self._pref_range(me, tg, override)
        bx, by = tg["pos"][0] - x, tg["pos"][1] - y
        bd = math.hypot(bx, by) + 1e-9
        ux, uy = bx / bd, by / bd
        if now >= self.strafe_until:
            if self.rng.random() < 0.6:
                self.strafe = -self.strafe
            self.strafe_until = now + 0.3 + self.rng.random() * 0.7
        sx, sy = -uy * self.strafe, ux * self.strafe
        tc = self._center(tg)
        cube, cube_d = None, 40.0
        for pos, _k, _s in self.b.mem["c"].values():
            cd = math.hypot(pos[0] - x, pos[1] - y)
            if cd < cube_d and abs(pos[2] - zf) < 4:
                cube, cube_d = pos, cd
        z = obs["zone"]
        best, bdx, bdy = -INF, 0.0, 0.0
        side_blocked = False
        for k in range(-1, 16):
            if k < 0:
                cx, cy, dx, dy, r = x, y, 0.0, 0.0, 0.0
            else:
                a = k * math.pi / 8
                r = 6.0
                dx, dy = math.cos(a), math.sin(a)
                cx, cy = x + dx * r, y + dy * r
                if w.wall_ahead(x, y, zf, a, r):
                    if dx * sx + dy * sy > 0.9:
                        side_blocked = True
                    continue
            cz = w.ground(cx, cy)
            if k >= 0 and abs(cz - zf) > 3.0:
                continue
            d = math.hypot(tg["pos"][0] - cx, tg["pos"][1] - cy)
            s = -1.1 * abs(d - rp) - r * 0.4
            los = w.clear_line([cx, cy, cz + mh], tc)
            if los:
                s += 45 * want_los
            body = [cx, cy, cz + body_h]
            for e in seen:
                if e is tg or math.hypot(e["pos"][0] - cx, e["pos"][1] - cy) > 170:
                    continue
                if w.clear_line(self._eye(e), body):
                    s -= 30
            s -= self._zone_penalty(z, cx, cy, now)
            if cube is not None:
                s += 1.5 * (cube_d - math.hypot(cube[0] - cx, cube[1] - cy))
            if want_los > 0:
                s += 9 * (dx * sx + dy * sy)
            if k < 0:
                s -= 8
                if want_los < 0 and not los:
                    s += 14
            if s > best:
                best, bdx, bdy = s, dx, dy
        if side_blocked:
            self.strafe = -self.strafe
            self.strafe_until = now + 0.5
        if bdx == 0.0 and bdy == 0.0:
            return None, 0.0
        return math.atan2(bdy, bdx), 1.0

    def _cover_move(self, obs: Dict[str, Any], me: Dict[str, Any], seen: List[Dict[str, Any]], now: float,
                    extra: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Optional[float], float]]:
        """Walk to the nearest reachable spot none of the (up to four nearest) threats can see. None if there is none."""
        w, nav = self.b.world, self.b.nav
        if nav is None:
            return None
        x, y, _zf = me["pos"]
        hgt = float(me.get("stats", {}).get("height", 9))
        threats = [t for t in seen if math.hypot(t["pos"][0] - x, t["pos"][1] - y) < 230]
        if extra is not None:
            threats.append(extra)
        if not threats:
            return None
        threats.sort(key=lambda t: math.hypot(t["pos"][0] - x, t["pos"][1] - y))
        threats = threats[:4]
        eyes = [self._eye(t) for t in threats]

        def hidden(cx: float, cy: float) -> bool:
            cz = w.ground(cx, cy)
            for e in eyes:
                if w.clear_line(e, [cx, cy, cz + hgt]) or w.clear_line(e, [cx, cy, cz + hgt * 0.5]):
                    return False
            return True

        if self.cover_ok and now - self.cover_at < 0.4 and self.cover_goal is not None:
            return self._to_cover(me, now)
        if not self.cover_ok and now - self.cover_at < 0.5:
            return None  # the last search found nothing; looking again every decision is costly and no likelier
        self.cover_at = now
        z = obs["zone"]
        if hidden(x, y) and self._zone_penalty(z, x, y, now) == 0:
            self.cover_ok, self.cover_goal = True, [x, y]
            return None, 0.0
        start = nav.nearest_free(*nav.to_cell(x, y))
        if start is None:
            self.cover_ok = False
            return None
        cell = nav.cell
        nx, blocked, ground, rise = nav.nx, nav.blocked, nav.ground, nav.max_rise
        s0 = start[1] * nx + start[0]
        steps = {s0: 0}
        queue = [s0]
        best_c, best_s = -1, INF
        checked = 0
        head = 0
        while head < len(queue) and checked < 40:
            c = queue[head]
            head += 1
            n = steps[c]
            dist_m = n * cell
            if dist_m > 100 or (best_c >= 0 and dist_m > best_s + 30):
                break
            ci, cj = c % nx, c // nx
            if (ci + cj) % 2 == 0 and n > 0:
                cx, cy = nav.center(ci, cj)
                checked += 1
                if hidden(cx, cy):
                    near = min(math.hypot(t["pos"][0] - cx, t["pos"][1] - cy) for t in threats)
                    sc = dist_m + self._zone_penalty(z, cx, cy, now) - min(near, 80.0) * 0.25
                    if sc < best_s:
                        best_c, best_s = c, sc
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                i, j = ci + di, cj + dj
                if i < 0 or j < 0 or i >= nx or j >= nav.ny:
                    continue
                nb = j * nx + i
                if nb in steps or blocked[nb] or ground[nb] - ground[c] > rise * (1.42 if di and dj else 1.0):
                    continue
                if di and dj and (blocked[cj * nx + i] or blocked[j * nx + ci]):
                    continue
                steps[nb] = n + 1
                queue.append(nb)
        if best_c < 0:
            self.cover_ok = False
            return None
        gx, gy = nav.center(best_c % nx, best_c // nx)
        self.cover_ok, self.cover_goal = True, [gx, gy]
        return self._to_cover(me, now)

    def _to_cover(self, me: Dict[str, Any], now: float) -> Tuple[Optional[float], float]:
        g = self.cover_goal
        if math.hypot(g[0] - me["pos"][0], g[1] - me["pos"][1]) < 1.5:
            return None, 0.0
        return self._travel(me, g[0], g[1], now)

    # ------------------------------------------------------------------ loot

    def _want_pickup(self, me: Dict[str, Any], kind: str) -> bool:
        ws = me["weapons"]
        if kind == "gun":
            return bool(ws) and ws[0] is None
        if any(w and w["kind"] == kind for w in ws) or kind == "mortar":
            return False
        cur = WEAPON_RANK.get(ws[1]["kind"], 0) if len(ws) > 1 and ws[1] else 0
        return WEAPON_RANK.get(kind, 0) > cur

    def _choose_loot(self, obs: Dict[str, Any], me: Dict[str, Any], hp_frac: float, danger: bool,
                     now: float) -> Optional[Tuple[str, Any, List[float]]]:
        w, nav = self.b.world, self.b.nav
        x, y = me["pos"][0], me["pos"][1]
        z = obs["zone"]
        cur = self._zone_now(z, now)
        walk = float(me.get("stats", {}).get("walk_speed", 11.7))
        cap = int(self.cfg.get("cubes", {}).get("stat_cap", 10))
        best: Optional[Tuple[str, Any, List[float]]] = None
        best_u = 0.0

        def consider(kind: str, iid: Any, pos: List[float], value: float) -> None:
            nonlocal best, best_u
            if self.bad.get(iid, 0) > now:
                return
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d > 260 or math.hypot(pos[0] - cur[0], pos[1] - cur[1]) > cur[2] - 6:
                return
            if now + d * 1.4 / walk + 5 > self._zone_deadline(z, pos[0], pos[1], 6):
                return
            if pos[2] - w.ground(pos[0], pos[1]) > 3.2:
                return
            if nav is not None and nav.nearest_free(*nav.to_cell(pos[0], pos[1]), max_r=2) is None:
                return
            u = value / (d + 18)
            if iid == self.loot_id:
                u *= 1.35
            if u > best_u:
                best_u, best = u, (kind, iid, pos)

        empty = len(me["weapons"]) > 1 and me["weapons"][1] is None
        for pid, (pos, kind, _s) in self.b.mem["p"].items():
            rad = self.loot_radius
            seek = kind == "laser"
            if seek:
                rad = self.laser_seek
            if self._want_pickup(me, kind) and math.hypot(pos[0] - x, pos[1] - y) < rad:
                v = 130.0 if empty else 60.0
                if seek:
                    v = max(v, 150.0)
                consider("pickup", pid, pos, v)
        cube_v = 25.0 if me.get("cubes", 0) >= cap else 75.0
        for cid, (pos, _k, _s) in self.b.mem["c"].items():
            if math.hypot(pos[0] - x, pos[1] - y) < max(60.0, self.loot_radius * 0.7):
                consider("cube", cid, pos, cube_v)
        if hp_frac > 0.5 and not danger:
            for kid, (pos, _k, _s) in self.b.mem["k"].items():
                if math.hypot(pos[0] - x, pos[1] - y) < self.loot_radius:
                    consider("crate", kid, pos, cube_v * 0.6)
            if me.get("cubes", 0) < cap:
                for nid, n in self.np.items():
                    if now - n["seen"] < 12 and math.hypot(n["pos"][0] - x, n["pos"][1] - y) < self.loot_radius:
                        consider("npc", nid, n["pos"], 85.0)
        if best is None:
            self.loot_id = None
            return None
        if best[1] != self.loot_id:
            self.loot_id, self.loot_since = best[1], now
        elif now - self.loot_since > 30:
            self.bad[best[1]] = now + 60
            self.loot_id = None
            return None
        return best

    # ------------------------------------------------------------------ aiming

    def _lead(self, muzzle: List[float], tp: List[float], tv: List[float], speed: float, grounded: bool) -> Tuple[float, float]:
        t = 0.0
        px, py, pz = tp
        for _ in range(4):
            tt = self.lat + t
            px, py, pz = tp[0] + tv[0] * tt, tp[1] + tv[1] * tt, tp[2]
            if not grounded:
                pz += tv[2] * tt * 0.5
            d = math.dist(muzzle, [px, py, pz])
            t = max(0.0, d - 3) / max(1.0, speed)
        return math.atan2(py - muzzle[1], px - muzzle[0]), math.atan2(pz - muzzle[2], math.hypot(px - muzzle[0], py - muzzle[1]))

    # ------------------------------------------------------------------ one decision

    def decide(self, obs: Dict[str, Any], me: Dict[str, Any], now: float) -> Dict[str, Any]:
        b, w = self.b, self.b.world
        self._observe(obs, me, now)
        st = me.get("stats", {})
        x, y, zf = me["pos"]
        z = obs["zone"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        out: Dict[str, Any] = {}
        if obs.get("offer") and obs["offer"].get("choices"):
            ranks = [UPGRADE_ORDER.index(c.get("id")) if c.get("id") in UPGRADE_ORDER else 99 for c in obs["offer"]["choices"]]
            out["pick"] = ranks.index(min(ranks))

        gi, gun = self._weapon(me, "gun")
        _li, light = self._weapon(me, "lightning")
        _zi, laser = self._weapon(me, "laser")
        late = float(z["next_radius"]) <= 70 or int(obs.get("alive", 16)) <= 3

        # ---------------- engage: which tracks are coming for us, and which one do we shoot
        seen: List[Dict[str, Any]] = []
        target = None
        best_s = INF
        for tk in self.tr.values():
            if now - tk["seen"] > 2.5:
                continue
            tk["engage"] = False
            seen.append(tk)
            if tk["seen"] != now:
                tk["los"] = False
                continue
            d = tk["dist"]
            to_me = math.atan2(y - tk["pos"][1], x - tk["pos"][0])
            aimed = abs(wrap(tk["aim"][0] - to_me)) < 0.22
            tk["engage"] = d < self.engage_dist or now - tk["hit"] < 6 or late or (tk["closing"] > 3 and d < 180) or \
                (tk["hp"] < 0.4 * tk["max_hp"] and d < 160) or (aimed and d < 190)
            if not tk["engage"] or not tk["los"] or d > self.max_target:
                continue
            s = d + 0.18 * tk["hp"]
            if tk["hp"] <= 0.3 * tk["max_hp"]:
                s -= 200
            if tk["id"] == self.stick_id and now - self.stick_at < 2:
                s -= 200
            if now - tk["hit"] < 4:
                s -= 40
            if tk["closing"] > 3:
                s -= 15
            if s < best_s:
                best_s, target = s, tk
        if target is not None:
            self.stick_id, self.stick_at = target["id"], now
        danger = False
        crowd = near = 0
        for tk in seen:
            if tk["engage"] or now - tk["hit"] < 4:
                danger = True
            if tk["seen"] == now:
                if tk["engage"] or (tk["los"] and tk["dist"] < 150):
                    crowd += 1
                if tk["dist"] < 160:
                    near += 1
        if danger or now - self.last_hurt < 3:
            self.last_threat = now

        # ---------------- posture: step back only while clearly losing, and only for a moment
        finishing = target is not None and target["dist"] < 120 and (target["hp"] < 200 or target["hp"] < me["hp"] * 0.5)
        losing = (near >= 2 and hp_frac < 0.55) or (near >= 3 and hp_frac < 0.8)
        if target is not None:
            tf = target["hp"] / max(1.0, target["max_hp"])
            if hp_frac < 0.35 and tf > hp_frac + 0.1:
                losing = True
            if tf < 0.25 and hp_frac > 0.2 and near <= 2:
                losing = False
        if losing:
            self.lose_at = now
        was = self.retreat
        self.retreat = now - self.lose_at < 1.5
        if self.retreat and not was:
            self.cover_at = -99.0

        # ---------------- move
        zgx, zgy = self._zone_goal(z, x, y)
        zdist = math.hypot(zgx - x, zgy - y)
        walk = float(st.get("walk_speed", 11.7))
        t_need = zdist * 1.45 / walk + 4
        deadline = self._zone_deadline(z, x, y, 8)
        zone_must = (not me["in_zone"]) or now + t_need + 3 >= deadline
        zone_should = now + t_need + 14 >= deadline
        energy = me["energy"]
        yaw: Optional[float] = None
        mag = 0.0
        boost = False
        mode = "roam"
        farm: Optional[Dict[str, Any]] = None
        farm_pos: Optional[List[float]] = None

        def fight(tg: Dict[str, Any]) -> None:
            nonlocal yaw, mag, boost, mode
            d = tg["dist"]
            ready = gun is not None and gun.get("ammo", 0) > 0 and gun.get("reload", 0) <= 0.25 and \
                d <= gun["stats"].get("range", 230) - 10
            if light is not None or laser is not None:
                ready = True
            want = 1.0 if ready else -1.0
            override = None
            if tg["hp"] < 0.38 * tg["max_hp"] and hp_frac > 0.42 and crowd <= 2:
                override = 17.0  # badly hurt and nobody else on us: run it down
            yaw, mag = self._fight_move(obs, me, tg, seen, want, now, override)
            if light is not None and d > 50 and energy > 55 and want > 0:
                boost = True
            mode = "fight"
            reach = 70.0 if hp_frac > 0.5 else 40.0
            cube = None
            for pos, _k, _s in b.mem["c"].values():
                cd = math.hypot(pos[0] - x, pos[1] - y)
                if cd < reach:
                    reach, cube = cd, pos
            if cube is not None:
                yaw, mag = self._travel(me, cube[0], cube[1], now)
                mode = "cube"

        if zone_must and zdist > 2:
            yaw, mag = self._travel(me, zgx, zgy, now, zone=True)
            boost = ((not me["in_zone"]) or now + t_need >= deadline) and energy > 5
            mode = "zone"
        elif self.retreat and not finishing and not late:
            cov = self._cover_move(obs, me, seen, now)
            if cov is not None:
                yaw, mag = cov
                boost = energy > 10 and mag > 0
                mode = "retreat"
            elif target is not None:
                yaw, mag = self._fight_move(obs, me, target, seen, 1.0, now)
                mode = "fight"
            else:
                yaw, mag = self._travel(me, zgx, zgy, now)
                mode = "roam"
        elif target is not None:
            fight(target)
        elif zone_should and zdist > 2:
            yaw, mag = self._travel(me, zgx, zgy, now, zone=True)
            mode = "zone"
        else:
            lc = self._choose_loot(obs, me, hp_frac, danger, now)
            if lc is not None:
                kind, iid, pos = lc
                mode = "loot"
                d = math.hypot(pos[0] - x, pos[1] - y)
                if kind == "npc":
                    n = self.np.get(iid)
                    if n is not None and now - n["seen"] < 1 and n["los"] and n["dist"] < 75:
                        farm = n
                        yaw, mag = self._fight_move(obs, me, n, seen, 1.0, now)
                        mode = "fight"
                    else:
                        yaw, mag = self._travel(me, pos[0], pos[1], now)
                elif kind == "crate":
                    cp = [pos[0], pos[1], pos[2] + float(self.cfg.get("crates", {}).get("height", 2.4)) * 0.5]
                    if d < 50 and w.clear_line([x, y, zf + float(st.get("muzzle_height", 7))], cp):
                        farm_pos = cp
                        if d < 12:
                            yaw, mag = math.atan2(y - pos[1], x - pos[0]), 1.0
                    else:
                        yaw, mag = self._travel(me, pos[0], pos[1], now)
                else:
                    yaw, mag = self._travel(me, pos[0], pos[1], now)
            elif zdist > 3:
                yaw, mag = self._travel(me, zgx, zgy, now)
                mode = "roam"

        # ---------------- sidestep bullets about to hit and the line of a laser being charged at us
        sdx = sdy = 0.0
        beam = False
        cz = zf + float(st.get("height", 9)) * 0.56
        reach = float(st.get("radius", 2.5)) + 1.1
        for p in obs.get("projectiles", []):
            if p.get("kind") == "mortar":
                continue
            vx, vy, vz = p["vel"]
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1:
                continue
            tc = ((x - p["pos"][0]) * vx + (y - p["pos"][1]) * vy + (cz - p["pos"][2]) * vz) / v2
            if tc <= 0 or tc > 0.7:
                continue
            mx, my = p["pos"][0] + vx * tc - x, p["pos"][1] + vy * tc - y
            vh = math.hypot(vx, vy)
            if math.hypot(mx, my) < reach and vh >= 1:
                px, py = -vy / vh, vx / vh
                if px * mx + py * my > 0:
                    px, py = -px, -py
                sdx += px
                sdy += py
        for m in obs["mechs"]:
            if m.get("kind") != "mech" or not m.get("charging") or m["id"] == me["id"]:
                continue
            to_me = math.atan2(y - m["pos"][1], x - m["pos"][0])
            if abs(wrap(m["aim"][0] - to_me)) > 0.15:
                continue
            sdx += math.cos(to_me + math.pi / 2 * self.strafe) * 1.5
            sdy += math.sin(to_me + math.pi / 2 * self.strafe) * 1.5
            beam = True
        sn = math.hypot(sdx, sdy)
        if sn > 0.01:
            mvx = math.cos(yaw) * mag * 0.4 if yaw is not None else 0.0
            mvy = math.sin(yaw) * mag * 0.4 if yaw is not None else 0.0
            mvx, mvy = mvx + sdx / sn * 0.9, mvy + sdy / sn * 0.9
            yaw, mag = math.atan2(mvy, mvx), min(1.0, math.hypot(mvx, mvy))
            if beam and energy > 25:
                boost = True
        out.update(move_yaw=yaw if mag > 0 else None, move_mag=mag, boost=bool(boost and mag > 0), mode=mode)

        # ---------------- shoot
        lat = self.lat
        mh = float(st.get("muzzle_height", 7))
        muzzle = [x + me["vel"][0] * lat, y + me["vel"][1] * lat, zf + mh]
        my_aim = me["aim"]
        fire = [False, False]
        aim = [my_aim[0], 0.0]
        have_aim = False
        defend = False
        if target is None and now - self.npc_hurt < 3:
            worst, best_off = None, 0.7
            for n in self.np.values():
                if now - n["seen"] > 0.5 or not n["los"] or n["dist"] > 150:
                    continue
                off = abs(wrap(math.atan2(n["pos"][1] - y, n["pos"][0] - x) - self.npc_bearing))
                if now - n["hit"] < 3:
                    off = 0.0
                if off < best_off:
                    best_off, worst = off, n
            if worst is not None:
                farm, farm_pos, defend = worst, None, True
        tp = tv = None
        tr = th = 0.0
        is_mech = False
        if target is not None:
            tr, th = b._size(target)
            tp, tv = list(target["pos"]), target["vel"]
            tp[2] += th * 0.5
            if not w.clear_line(muzzle, tp):
                tp[2] = target["pos"][2] + th * 0.85
            is_mech = True
        elif farm is not None:
            tr, th = b._size(farm)
            tp, tv = list(farm["pos"]), farm["vel"]
            tp[2] += th * 0.5
        elif farm_pos is not None:
            cc = self.cfg.get("crates", {})
            tr, th = float(cc.get("radius", 1.4)), float(cc.get("height", 2.4))
            tp, tv = list(farm_pos), [0.0, 0.0, 0.0]
        if tp is not None and not is_mech and (self.retreat or hp_frac < 0.5) and not defend:
            tp = None  # hurt: let the regeneration clock run instead of resetting it on loot
        if tp is not None:
            dist = math.hypot(tp[0] - x, tp[1] - y)
            grounded = abs(tv[2]) < 0.5
            gy = gp = 0.0
            gun_ok = False
            if gun is not None:
                gy, gp = self._lead(muzzle, tp, tv, float(gun["stats"].get("speed", 252)), grounded)
                gun_ok = dist <= float(gun["stats"].get("range", 230)) * 0.96
            dpx, dpy = tp[0] + tv[0] * (lat + 0.05), tp[1] + tv[1] * (lat + 0.05)
            hy = math.atan2(dpy - muzzle[1], dpx - muzzle[0])
            hp_ = math.atan2(tp[2] - muzzle[2], math.hypot(dpx - muzzle[0], dpy - muzzle[1]))
            aim = [gy, gp] if gun is not None else [hy, hp_]
            have_aim = True
            tol_y = math.atan2(tr * 0.75, max(dist, 1.0))
            tol_p = math.atan2(th * 0.4, max(dist, 1.0))
            turn = float(st.get("turn_rate", 4)) * 0.05
            pturn = float(st.get("pitch_rate", 3)) * 0.05

            def slew(wy: float, wp: float) -> bool:
                return max(0.0, abs(wrap(wy - my_aim[0])) - turn) < tol_y and max(0.0, abs(wp - my_aim[1]) - pturn) < tol_p

            for i, wpn in enumerate(me["weapons"][:2]):
                if wpn is None:
                    continue
                kind = wpn["kind"]
                ws = wpn["stats"]
                if kind == "gun":
                    if gun_ok and wpn.get("ammo", 0) > 0 and wpn.get("reload", 0) <= 0 and slew(gy, gp):
                        fire[i] = True
                elif kind == "lightning":
                    cone = math.radians(float(ws.get("cone_deg", 30)))
                    if dist < float(ws.get("range", 45)) - 1.5 and abs(wrap(hy - my_aim[0])) < cone * 0.5 - 0.05 and \
                            energy >= float(ws.get("energy", 3)):
                        fire[i] = True
                elif kind == "laser":
                    if not is_mech or dist > float(ws.get("range", 280)) * 0.97:
                        continue
                    full = float(ws.get("charge", 0.75))
                    charge = float(wpn.get("charge", 0.0))
                    on_line = abs(wrap(hy - my_aim[0])) < tol_y * 0.9 and abs(hp_ - my_aim[1]) < tol_p
                    if float(ws.get("min_charge", 0.15)) <= charge < full - 0.051:
                        mf = float(ws.get("min_frac", 0.35))
                        cc = self.cfg.get("cubes", {})
                        dmg = float(ws.get("damage", 315)) * (mf + (1 - mf) * charge / full) * \
                            (1 + float(cc.get("damage_per_cube", 0.04)) * min(float(me.get("cubes", 0)), float(cc.get("stat_cap", 10))))
                        if "mortar" in target.get("weapons", []):
                            dmg *= 1 + float(ws.get("cookoff", 0.6))
                        if dmg * 0.7 >= target["hp"] and on_line:
                            aim = [hy, hp_]
                            fire[i] = False  # what is charged already kills: let go
                            continue
                    if charge < full - 0.051:
                        fire[i] = True
                        if charge > full * 0.45:
                            aim = [hy, hp_]
                    else:
                        aim = [hy, hp_]
                        fire[i] = not on_line
        if not have_aim:
            watch = None
            for tk in seen:
                if (tk["engage"] or now - tk["hit"] < 6) and (watch is None or tk["dist"] < watch["dist"]):
                    watch = tk
            if watch is not None:
                aim[0] = math.atan2(watch["pos"][1] - y, watch["pos"][0] - x)
            elif now - self.npc_hurt < 2.5:
                aim[0] = self.npc_bearing
            elif now - self.last_hurt < 2.5:
                aim[0] = self.hurt_bearing
            elif out["move_yaw"] is not None:
                aim[0] = out["move_yaw"] + 1.15 * math.sin(now * 1.3)
            else:
                aim[0] = my_aim[0] + 0.35
            aim[1] = 0.0
        aim[1] = clamp(aim[1], float(st.get("min_pitch", -0.7)), float(st.get("max_pitch", 1.3)))
        out["aim"], out["fire"] = aim, fire
        out["reload"] = bool(gun is not None and gun.get("reload", 0) <= 0 and gun.get("ammo", 0) < gun["stats"].get("mag", 12)
                             and tp is None and (gun.get("ammo", 0) <= gun["stats"].get("mag", 12) * 0.6 or now - self.last_threat > 3))

        # ---------------- weapons within reach
        reach_p = float(self.cfg["loot"]["pickup_radius"]) - 0.3
        for p in obs.get("pickups", []):
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) > reach_p or abs(p["pos"][2] - zf) > 3.6:
                continue
            if not self._want_pickup(me, p["kind"]) or b.bad_pickups.get(p["id"], 0) > now:
                continue
            ws = me["weapons"]
            if len(ws) > 1 and ws[1] is not None and ws[0] is not None:
                if ws[1]["kind"] == "laser" and float(ws[1].get("charge", 0)) > 0:
                    break
                out["drop"] = 1
            out["take"] = p["id"]
            break
        return out
