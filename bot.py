"""Mech Royale bot.

Plan each decision (10 per second):
  1. Read the world: enemies, loot, zone, recent damage and sounds.
  2. Pick a mode: zone (get inside the safe circle), retreat (hide and regenerate),
     fight (hold a good range and strafe), loot (weapons, cubes, crates, drones, supply drops) or roam.
  3. Steer around buildings toward that mode's goal.
  4. Aim with lead for the gun, straight for the laser, an arc for the mortar, and fire only
     when the torso is actually on target and nothing solid is in the way.

Rules and numbers come from hello.config (protocol.md); the map comes from hello.map.dir.
"""
from __future__ import annotations

import math
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf

try:
    from world import World
except Exception:  # never let a map problem stop the bot from answering
    World = None  # type: ignore

TAU = 2 * math.pi

# Higher is better. The gun is the dependable middle-range weapon, so only the laser outranks it.
WEAPON_VALUE = {"laser": 3.0, "gun": 2.5, "lightning": 2.0, "mortar": 1.0}

# Upgrade preference, best first. Survival first (placement scores, not kills), then damage.
UPGRADE_PREF = [
    "armor", "plating", "nanites", "laser_damage", "laser_charge", "gun_damage", "gun_rate",
    "lightning_efficiency", "lightning_range", "gyros", "gun_mag", "mortar_reload", "servos",
    "sensors", "gun_velocity", "reactor", "capacitor", "jets", "lightning_chain",
    "mortar_velocity", "mortar_splash", "laser_pierce",
]


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


def dist2d(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(yaw: float) -> List[float]:
    return [math.cos(yaw), math.sin(yaw)]


class Bot:
    name = "royale-v1"

    def __init__(self, hello: Dict[str, Any]) -> None:
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(int(hello.get("seed", 0)) & 0xFFFFFFFF)
        self.world = None
        if World is not None:
            try:
                m = hello.get("map") or {}
                self.world = World(m.get("dir", "/arena"), m.get("files"))
            except Exception:
                mf.log("map load failed, playing without it:\n" + traceback.format_exc())
        self.reach = float(cfg["loot"]["pickup_radius"])
        self.cube_reach = float(cfg["cubes"].get("pickup_radius", 3.5))
        self.mech_r = float(cfg["mech"]["radius"])
        self.mech_h = float(cfg["mech"]["height"])
        self.size_per_cube = float(cfg["cubes"].get("size_per_cube", 0.05))
        self.npc_h = float(cfg["npc"].get("height", 3.5))
        self.npc_r = float(cfg["npc"].get("radius", 1.8))
        self.mortar_g = float(cfg["weapons"]["mortar"].get("gravity", 22))

        self.prev_t = 0.0
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.avoid_side = 1
        self.stuck_s = 0.0
        self.unstick_until = 0.0
        self.unstick_yaw = 0.0
        self.unstick_count = 0
        self.last_damage: Optional[Tuple[float, Optional[float], Any, str]] = None
        self.last_hurt_t = -99.0
        self.last_sound: Optional[Tuple[float, float, str]] = None
        self.retreating = False
        self.cover: Optional[List[float]] = None
        self.cover_t = -99.0
        self.target_id: Any = None
        self.wander: Optional[List[float]] = None
        self.wander_until = 0.0
        self.bad_pickups: Dict[Any, float] = {}
        self.threats: Dict[Any, Tuple[float, List[float]]] = {}
        self.slow_ticks = 0

    # ------------------------------------------------------------------ main

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        act = self._act(obs)
        ms = (time.perf_counter() - t0) * 1000
        if ms > 25:
            self.slow_ticks += 1
            if self.slow_ticks <= 5 or self.slow_ticks % 50 == 0:
                mf.log(f"slow decision {ms:.1f} ms at t={obs['t']:.1f} ({self.slow_ticks} slow so far)")
        return act

    def _act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        t = float(obs["t"])
        dt = max(0.0, min(0.5, t - self.prev_t))
        self.prev_t = t
        me = obs["self"]
        x, y = me["pos"][0], me["pos"][1]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act: Dict[str, Any] = {"tick": obs["tick"]}

        self._read_events(obs, t)
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for e in enemies:
            self.threats[e["id"]] = (t, e["pos"])
        self.threats = {k: v for k, v in self.threats.items() if t - v[0] < 8}

        weapons = me["weapons"]
        held = {w["kind"]: i for i, w in enumerate(weapons) if w}

        # ---------------- target choice
        target = self._choose_target(me, enemies)
        tgt_d = dist2d(target["pos"], me["pos"]) if target else 1e9
        tgt_los = self._los(me, target) if target else False

        # ---------------- mode choice
        zone_goal, zone_urgent = self._zone_goal(obs, me, t)
        near_enemies = [e for e in enemies if dist2d(e["pos"], me["pos"]) < 160]
        self._update_retreat(hp_frac, near_enemies, target, tgt_d, t)

        move_yaw: Optional[float] = None
        move_mag = 1.0
        boost = False
        mode = "roam"
        loot_target: Optional[Dict[str, Any]] = None  # crate or drone to shoot when no enemy

        if not me["in_zone"] or zone_urgent:
            mode = "zone"
            move_yaw = math.atan2(zone_goal[1] - y, zone_goal[0] - x)
            boost = (not me["in_zone"]) or me["energy"] > 60
        elif self.retreating and (target is not None or self.threats):
            # keep hiding from enemies we saw in the last few seconds, even once they drop out of view
            mode = "retreat"
            remembered = [{"pos": p} for (seen_t, p) in self.threats.values()]
            move_yaw, move_mag, boost = self._retreat_move(me, near_enemies or remembered, t, zone_goal)
        elif target is not None:
            mode = "fight"
            move_yaw, move_mag, boost = self._fight_move(me, target, tgt_d, tgt_los, held, enemies, t)
        else:
            goal, loot_target, why = self._loot_goal(obs, me, held, t, zone_goal, hp_frac, npcs)
            if goal is not None:
                mode = why
                d = dist2d(goal, me["pos"])
                move_yaw = math.atan2(goal[1] - y, goal[0] - x)
                move_mag = 1.0 if d > 3 else 0.3
            elif self.last_damage and t - self.last_damage[0] < 3 and self.last_damage[1] is not None:
                mode = "evade"
                move_yaw = self.last_damage[1] + math.pi / 2 * self.strafe_dir
                boost = me["energy"] > 40
                self._maybe_flip_strafe(t)
            else:
                mode = "roam"
                goal = self._wander_goal(obs, me, t, zone_goal)
                move_yaw = math.atan2(goal[1] - y, goal[0] - x)
                # walking slowly is silent (protocol.md section 8)
                move_mag = 0.4 if dist2d(goal, me["pos"]) < 120 else 1.0

        # ---------------- dodge a laser that is charging at us
        if target is not None and mode in ("fight", "retreat"):
            for e in enemies:
                if e.get("charging") and self._aimed_at_me(e, me):
                    bearing = math.atan2(e["pos"][1] - y, e["pos"][0] - x)
                    move_yaw = bearing + math.pi / 2 * self.strafe_dir
                    move_mag, boost = 1.0, me["energy"] > 15
                    break

        # ---------------- steering
        jump = False
        if move_yaw is not None:
            move_yaw, jump_needed = self._steer(me, move_yaw, t, dt, move_mag)
            jump = jump or jump_needed
            act["move"] = [round(math.cos(move_yaw) * move_mag, 3), round(math.sin(move_yaw) * move_mag, 3)]
        else:
            act["move"] = [0.0, 0.0]
        if t < self.unstick_until and not me["grounded"] and me["energy"] > 10:
            boost = True  # jet over the obstacle
        act["boost"] = bool(boost)
        act["jump"] = bool(jump and me["grounded"])

        # ---------------- aim and fire
        aim_obj = target if target is not None else loot_target
        aim, fire, reload = self._aim_and_fire(me, aim_obj, held, t, mode, move_yaw)
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire
        if reload:
            act["reload"] = True

        # ---------------- pickups in reach
        take = self._take_pickup(obs, me, held, weapons, t)
        if take:
            act.update(take)

        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{me['energy']:.0f} a{obs['alive']}")
        return act

    # ------------------------------------------------------------------ reading

    def _read_events(self, obs: Dict[str, Any], t: float) -> None:
        for e in obs.get("events", []):
            et = e.get("type")
            if et == "damaged":
                if e.get("weapon") != "zone":
                    self.last_damage = (t, e.get("bearing"), e.get("by"), e.get("weapon", ""))
                    self.last_hurt_t = t
            elif et == "take_failed":
                self.bad_pickups[e.get("id")] = t + 6.0
        best = None
        for s in obs.get("sounds", []):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and s.get("dist") in ("near", "mid"):
                if best is None or s.get("age", 0) < best.get("age", 0):
                    best = s
        if best is not None:
            self.last_sound = (t - float(best.get("age", 0)), float(best["bearing"]), best["kind"])

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        choices = offer.get("choices") or []
        best_i, best_rank = 0, 1e9
        for i, c in enumerate(choices):
            cid = c.get("id", "")
            rank = UPGRADE_PREF.index(cid) if cid in UPGRADE_PREF else 50
            if rank < best_rank:
                best_i, best_rank = i, rank
        return best_i

    # ------------------------------------------------------------------ geometry helpers

    def _size(self, m: Dict[str, Any]) -> Tuple[float, float]:
        """(radius, height) of a mech or drone."""
        if m.get("kind") == "npc":
            return self.npc_r, self.npc_h
        s = 1 + self.size_per_cube * float(m.get("cubes", 0) or 0)
        return self.mech_r * s, self.mech_h * s

    def _muzzle(self, me: Dict[str, Any]) -> List[float]:
        st = me.get("stats", {})
        return [me["pos"][0], me["pos"][1], me["pos"][2] + float(st.get("muzzle_height", 7))]

    def _los(self, me: Dict[str, Any], m: Dict[str, Any]) -> bool:
        if self.world is None:
            return True
        _, h = self._size(m)
        p = m["pos"]
        a = self._muzzle(me)
        return self.world.clear_line(a, [p[0], p[1], p[2] + h * 0.5]) or \
            self.world.clear_line(a, [p[0], p[1], p[2] + h * 0.85])

    def _aimed_at_me(self, e: Dict[str, Any], me: Dict[str, Any]) -> bool:
        d = max(1.0, dist2d(e["pos"], me["pos"]))
        bearing = math.atan2(me["pos"][1] - e["pos"][1], me["pos"][0] - e["pos"][0])
        return abs(wrap(e["aim"][0] - bearing)) < 0.06 + 6.0 / d

    def _maybe_flip_strafe(self, t: float) -> None:
        if t >= self.strafe_until:
            self.strafe_dir = -self.strafe_dir if self.rng.random() < 0.7 else self.strafe_dir
            self.strafe_until = t + self.rng.uniform(0.8, 2.2)

    # ------------------------------------------------------------------ targets and modes

    def _choose_target(self, me: Dict[str, Any], enemies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best, best_s = None, 1e9
        for e in enemies:
            d = dist2d(e["pos"], me["pos"])
            hpf = float(e.get("hp", 1)) / max(1.0, float(e.get("max_hp", 1) or 1))
            s = d + 70 * hpf
            if not self._los(me, e):
                s += 120
            if e["id"] == self.target_id:
                s -= 35
            if s < best_s:
                best, best_s = e, s
        self.target_id = best["id"] if best else None
        return best

    def _zone_goal(self, obs: Dict[str, Any], me: Dict[str, Any], t: float) -> Tuple[List[float], bool]:
        zn = obs["zone"]
        nc, nr = zn["next_center"], float(zn["next_radius"])
        d = dist2d(me["pos"], nc)
        margin = min(25.0, nr * 0.25)
        need = d - (nr - margin)
        if need <= 0:
            return nc, False
        speed = float(me.get("stats", {}).get("walk_speed", 11.7))
        time_left = float(zn["shrink_end"]) - t
        urgent = need / speed + 15 > time_left
        return nc, urgent

    def _update_retreat(self, hp_frac: float, near: List[Dict[str, Any]], target, tgt_d: float, t: float) -> None:
        many = len(near) >= 2
        # a full laser hit does ~315 (more with cubes): leave before one more hit would leave us helpless
        laser_near = any("laser" in (e.get("weapons") or []) for e in near)
        if not self.retreating:
            if hp_frac < 0.38 or (many and hp_frac < 0.6) or (laser_near and hp_frac < 0.45):
                # finish an almost-dead target instead of turning away
                if not (target is not None and float(target.get("hp", 9999)) < 160 and tgt_d < 120):
                    self.retreating = True
                    self.cover = None
        else:
            calm = t - self.last_hurt_t > 8
            if hp_frac > 0.85 or (calm and hp_frac > 0.6 and not many):
                self.retreating = False

    def _fight_move(self, me, target, d, los, held, enemies, t) -> Tuple[float, float, bool]:
        x, y = me["pos"][0], me["pos"][1]
        to_t = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
        energy = me["energy"]
        if "lightning" in held and d < 70 and energy > 25:
            lo, hi = 8.0, 28.0
        elif "laser" in held:
            lo, hi = 60.0, 140.0
        else:
            lo, hi = 40.0, 100.0
        if not los:
            lo, hi = 0.0, min(hi, max(20.0, d - 25))  # close in to regain a line of fire
        self._maybe_flip_strafe(t)
        radial = 0.0
        if d > hi:
            radial = 1.0
        elif d < lo:
            radial = -1.0
        side = to_t + math.pi / 2 * self.strafe_dir
        vx = radial * math.cos(to_t) + math.cos(side) * (0.8 if radial else 1.0)
        vy = radial * math.sin(to_t) + math.sin(side) * (0.8 if radial else 1.0)
        boost = (radial > 0 and d > hi + 30 and energy > 55) or (radial < 0 and energy > 35)
        return math.atan2(vy, vx), 1.0, boost

    def _retreat_move(self, me, threats, t, zone_goal) -> Tuple[float, float, bool]:
        x, y = me["pos"][0], me["pos"][1]
        if self.cover is None or t - self.cover_t > 1.5:
            self.cover = self._find_cover(me, threats, zone_goal)
            self.cover_t = t
        if self.cover is not None:
            d = dist2d(self.cover, me["pos"])
            if d < 2.5:
                return math.atan2(self.cover[1] - y, self.cover[0] - x), 0.0, False
            return math.atan2(self.cover[1] - y, self.cover[0] - x), 1.0, me["energy"] > 20
        # no cover found: run directly away from the threats, staying toward the zone
        ax = ay = 0.0
        for e in threats:
            a = math.atan2(y - e["pos"][1], x - e["pos"][0])
            ax += math.cos(a)
            ay += math.sin(a)
        zy = math.atan2(zone_goal[1] - y, zone_goal[0] - x)
        ax += 0.4 * math.cos(zy)
        ay += 0.4 * math.sin(zy)
        self._maybe_flip_strafe(t)
        away = math.atan2(ay, ax) + 0.5 * self.strafe_dir
        return away, 1.0, me["energy"] > 20

    def _find_cover(self, me, threats, zone_goal) -> Optional[List[float]]:
        w = self.world
        if w is None:
            return None
        x, y, z = me["pos"]
        threats = sorted(threats, key=lambda e: dist2d(e["pos"], me["pos"]))[:2]
        eyes = [[e["pos"][0], e["pos"][1], e["pos"][2] + 8.0] for e in threats]

        def hidden(cx: float, cy: float, cz: float) -> bool:
            for ex in eyes:
                if not (w.building_between(ex, [cx, cy, cz + 4.5]) and w.building_between(ex, [cx, cy, cz + 8.5])):
                    return False
            return True

        if hidden(x, y, z):
            return [x, y, z]
        best, best_c = None, 1e9
        for r in (12.0, 22.0, 34.0):
            for k in range(16):
                a = k * TAU / 16
                cx, cy = x + r * math.cos(a), y + r * math.sin(a)
                cz = w.ground(cx, cy)
                if cz - z > 2.0 or w.inside_solid(cx, cy, cz + 1.5) or w.inside_solid(cx, cy, cz + 6):
                    continue
                if w.building_between([x, y, z + 1.6], [cx, cy, cz + 1.6]):
                    continue  # keep to straight, walkable paths
                if not hidden(cx, cy, cz):
                    continue
                cost = r
                for e in threats:
                    toward = math.cos(a - math.atan2(e["pos"][1] - y, e["pos"][0] - x))
                    cost += 25 * max(0.0, toward)
                if dist2d([cx, cy], zone_goal) > dist2d(me["pos"], zone_goal) + 10:
                    cost += 10
                if cost < best_c:
                    best, best_c = [cx, cy, cz], cost
        return best

    def _loot_goal(self, obs, me, held, t, zone_goal, hp_frac, npcs):
        """Where to walk when no enemy mech is in view, plus a crate or drone to shoot on the way."""
        pos = me["pos"]
        z = pos[2]
        worst = self._worst_slot_value(me)
        best, best_c, why = None, 1e9, "loot"
        for p in obs.get("pickups", []):
            if self.bad_pickups.get(p["id"], 0) > t or abs(p["pos"][2] - z) > 3.5:
                continue
            if p["kind"] in held or WEAPON_VALUE.get(p["kind"], 0) <= worst:
                continue
            c = dist2d(p["pos"], pos) - 40 * (WEAPON_VALUE[p["kind"]] - worst)
            if dist2d(p["pos"], pos) < 150 and c < best_c:
                best, best_c, why = p["pos"], c, "weapon"
        for c in obs.get("cubes", []):
            d = dist2d(c["pos"], pos)
            if d < 90 and abs(c["pos"][2] - z) < 3.5 and d - 20 < best_c:
                best, best_c, why = c["pos"], d - 20, "cube"
        for dr in obs.get("drops", []):
            d = dist2d(dr["pos"], pos)
            speed = float(me.get("stats", {}).get("walk_speed", 11.7))
            if d < 220 and float(dr.get("lands_in", 0)) < d / speed + 8 and dr.get("holder") in (None, me["id"]):
                c = d - 60
                if c < best_c:
                    best, best_c, why = dr["pos"], c, "drop"
        # something to shoot: a crate (cube + maybe a weapon) or, early on, a drone (2 cubes)
        shoot = None
        sd = 1e9
        for cr in obs.get("crates", []):
            d = dist2d(cr["pos"], pos)
            if d < 130 and d < sd and self._clear_to(me, [cr["pos"][0], cr["pos"][1], cr["pos"][2] + 0.8]):
                shoot, sd = {"kind": "crate", "pos": cr["pos"], "vel": [0, 0, 0], "id": cr["id"]}, d
        if t < 240 and hp_frac > 0.6:
            for n in npcs:
                d = dist2d(n["pos"], pos)
                if d < 120 and d < sd and self._los(me, n):
                    shoot, sd = n, d
        return best, shoot, why

    def _clear_to(self, me, point) -> bool:
        return self.world is None or self.world.clear_line(self._muzzle(me), point)

    def _wander_goal(self, obs, me, t, zone_goal) -> List[float]:
        zn = obs["zone"]
        nr = float(zn["next_radius"])
        if dist2d(me["pos"], zone_goal) > nr * 0.55:
            return zone_goal
        if self.wander is None or t > self.wander_until or dist2d(self.wander, me["pos"]) < 5:
            a = self.rng.uniform(0, TAU)
            r = self.rng.uniform(0, nr * 0.4)
            self.wander = [zone_goal[0] + r * math.cos(a), zone_goal[1] + r * math.sin(a)]
            self.wander_until = t + self.rng.uniform(8, 16)
        return self.wander

    def _worst_slot_value(self, me) -> float:
        vals = [WEAPON_VALUE.get(w["kind"], 0) if w else 0.0 for w in me["weapons"]]
        return min(vals) if vals else 0.0

    # ------------------------------------------------------------------ movement

    def _steer(self, me, yaw: float, t: float, dt: float, mag: float) -> Tuple[float, bool]:
        """Turn the wanted direction aside from walls; detect being stuck and jump/jet out."""
        x, y, z = me["pos"]
        speed = math.hypot(me["vel"][0], me["vel"][1])
        if mag > 0.3 and speed < 1.5 and me["grounded"]:
            self.stuck_s += dt
        else:
            self.stuck_s = max(0.0, self.stuck_s - dt)
        if self.stuck_s > 0.8:
            self.stuck_s = 0.0
            self.unstick_count += 1
            # alternate: first try climbing over (jump + jet forward), then a sidestep
            if self.unstick_count % 2 == 1:
                self.unstick_yaw = yaw
            else:
                self.unstick_yaw = yaw + self.rng.choice([-1, 1]) * self.rng.uniform(1.2, 2.2)
            self.unstick_until = t + 1.3
            return self.unstick_yaw, True
        if t < self.unstick_until:
            return self.unstick_yaw, False
        w = self.world
        if w is None or mag <= 0:
            return yaw, False
        probe = 7.0
        if not w.wall_ahead(x, y, z, yaw, probe):
            return yaw, False
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            for side in (self.avoid_side, -self.avoid_side):
                cand = yaw + side * k * 0.27
                if not w.wall_ahead(x, y, z, cand, probe):
                    self.avoid_side = side
                    return cand, False
        return yaw, True

    # ------------------------------------------------------------------ aiming and firing

    def _aim_and_fire(self, me, obj, held, t, mode, move_yaw) -> Tuple[List[float], List[bool], bool]:
        weapons = me["weapons"]
        fire = [False, False]
        reload = False
        my_aim = me["aim"]
        gun_i = held.get("gun")
        gun = weapons[gun_i] if gun_i is not None else None

        if obj is None:
            # nothing to shoot: keep the laser from wasting a charge, top up the gun, look around
            self._hold_laser(me, held, fire, release=mode == "retreat")
            if gun and 0 < gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.6 and not gun.get("reload"):
                reload = True
            return [self._scan_yaw(me, t, move_yaw), 0.0], fire, reload

        is_mech = obj.get("kind") == "mech"
        if obj.get("kind") == "crate":
            radius, height = 1.4, 1.6
        else:
            radius, height = self._size(obj)
        tgt = obj["pos"]
        vel = obj.get("vel", [0, 0, 0])
        center_z = tgt[2] + height * 0.5
        muzzle = self._muzzle(me)
        d3 = math.dist(muzzle, [tgt[0], tgt[1], center_z])
        los = self._clear_to(me, [tgt[0], tgt[1], center_z]) or (
            obj.get("kind") != "crate" and self._clear_to(me, [tgt[0], tgt[1], tgt[2] + height * 0.85]))

        laser_i = held.get("laser")
        light_i = held.get("lightning")
        mortar_i = held.get("mortar")
        laser = weapons[laser_i] if laser_i is not None else None
        light = weapons[light_i] if light_i is not None else None
        mortar = weapons[mortar_i] if mortar_i is not None else None

        # Retreating: keep shooting back whenever the line is clear (only our shots remove enemies),
        # but don't start a new laser charge, because charging halves walking speed.
        retreating = mode == "retreat"

        # primary weapon decides the aim this tick
        primary = "gun"
        if laser and los and d3 <= laser["stats"].get("range", 280) and \
                (laser.get("charge", 0) > 0 or (laser["cooldown"] < 0.3 and not retreating)):
            primary = "laser"
        elif mortar and is_mech and mortar["cooldown"] <= 0.1 and 25 < d3 < 220 and (not los or (gun and gun.get("reload", 0) > 0)):
            primary = "mortar"

        if primary == "laser":
            aim = self._direct_aim(muzzle, tgt, vel, center_z, None, lead_s=0.1)
        elif primary == "mortar":
            sol = self._mortar_aim(me, mortar, tgt, vel, los)
            aim = sol if sol else self._direct_aim(muzzle, tgt, vel, center_z, gun["stats"]["speed"] if gun else None)
            if sol is None:
                primary = "gun"
        else:
            speed = gun["stats"]["speed"] if gun else 250.0
            aim = self._direct_aim(muzzle, tgt, vel, center_z, speed)

        def on_target(want: List[float], tol: float) -> bool:
            ey = wrap(want[0] - my_aim[0])
            ep = want[1] - my_aim[1]
            return d3 * math.hypot(ey * math.cos(want[1]), ep) < tol

        # gun
        if gun_i is not None and gun and los and d3 <= gun["stats"].get("range", 230):
            if primary == "gun":
                if gun.get("ammo", 1) > 0 and on_target(aim, radius * 0.9 + 0.3):
                    fire[gun_i] = True
            else:
                gaim = self._direct_aim(muzzle, tgt, vel, center_z, gun["stats"]["speed"])
                if gun.get("ammo", 1) > 0 and on_target(gaim, radius * 0.8):
                    fire[gun_i] = True
        if gun and gun.get("ammo", 1) == 0 and not gun.get("reload"):
            reload = True

        # laser: hold to charge, let go when charged and on target
        if laser_i is not None and laser:
            full = laser["stats"].get("charge", 0.75)
            charge = laser.get("charge", 0.0)
            if primary == "laser":
                if laser["cooldown"] <= 0.05 and charge < full - 0.02:
                    fire[laser_i] = True
                elif charge > 0:
                    fire[laser_i] = not on_target(aim, radius * 0.7)
            else:
                self._hold_laser(me, held, fire, release=retreating)

        # lightning: fires at the nearest thing in its cone, so just check range and cone
        if light_i is not None and light and is_mech and los:
            rng = light["stats"].get("range", 45)
            cone = math.radians(light["stats"].get("cone_deg", 30)) * 0.5
            yaw_to = math.atan2(tgt[1] - muzzle[1], tgt[0] - muzzle[0])
            if dist2d(tgt, muzzle) < rng * 0.95 and abs(wrap(yaw_to - my_aim[0])) < cone \
                    and me["energy"] > light["stats"].get("energy", 3) + 8 and light["cooldown"] <= 0.05:
                fire[light_i] = True

        # mortar
        if mortar_i is not None and mortar and primary == "mortar" and mortar["cooldown"] <= 0.05:
            ey = abs(wrap(aim[0] - my_aim[0]))
            ep = abs(aim[1] - my_aim[1])
            if ey < 0.025 and ep < 0.025:
                fire[mortar_i] = True

        return aim, fire, reload

    def _hold_laser(self, me, held, fire, release: bool = False) -> None:
        """No shot for the laser right now. Keep a charge in progress for when the target reappears,
        unless `release` (retreating): then let it go so we stop walking at half speed."""
        laser_i = held.get("laser")
        if laser_i is None:
            return
        w = me["weapons"][laser_i]
        if w and w.get("charge", 0) > 0:
            fire[laser_i] = not release

    def _direct_aim(self, muzzle, tgt, vel, center_z, speed: Optional[float], lead_s: float = 0.0) -> List[float]:
        px, py, pz = tgt[0], tgt[1], center_z
        if speed:
            for _ in range(3):
                tt = math.dist(muzzle, [px, py, pz]) / speed
                px = tgt[0] + vel[0] * tt
                py = tgt[1] + vel[1] * tt
                pz = center_z + (vel[2] * tt if abs(vel[2]) > 0.5 else 0.0)
        elif lead_s:
            px += vel[0] * lead_s
            py += vel[1] * lead_s
        dx, dy, dz = px - muzzle[0], py - muzzle[1], pz - muzzle[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def _mortar_aim(self, me, mortar, tgt, vel, los) -> Optional[List[float]]:
        st = mortar["stats"]
        v = float(st.get("speed", 72))
        g = float(st.get("gravity", self.mortar_g))
        mst = me.get("stats", {})
        ox, oy = me["pos"][0], me["pos"][1]
        oz = me["pos"][2] + float(mst.get("height", 9)) + 0.5
        px, py, pz = tgt[0], tgt[1], tgt[2] + 1.0
        lo_p = float(self.cfg["mech"].get("min_pitch", -0.7))
        hi_p = float(self.cfg["mech"].get("max_pitch", 1.3))
        sol = None
        for _ in range(3):
            d = math.hypot(px - ox, py - oy)
            h = pz - oz
            disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
            if disc < 0 or d < 1:
                return None
            root = math.sqrt(disc)
            pitch = math.atan((v * v - root) / (g * d)) if los else math.atan((v * v + root) / (g * d))
            if not lo_p <= pitch <= hi_p:
                return None
            tt = d / (v * math.cos(pitch))
            px, py = tgt[0] + vel[0] * tt, tgt[1] + vel[1] * tt
            sol = [math.atan2(py - oy, px - ox), pitch]
        return sol

    def _scan_yaw(self, me, t, move_yaw) -> float:
        if self.last_damage and t - self.last_damage[0] < 4 and self.last_damage[1] is not None:
            return self.last_damage[1]
        if self.last_sound and t - self.last_sound[0] < 3:
            return self.last_sound[1]
        recent = [v for v in self.threats.values() if t - v[0] < 8]
        if recent:
            p = max(recent, key=lambda v: v[0])[1]
            return math.atan2(p[1] - me["pos"][1], p[0] - me["pos"][0])
        base = move_yaw if move_yaw is not None else me["aim"][0]
        return base + 1.0 * math.sin(t * 1.1)

    # ------------------------------------------------------------------ pickups

    def _take_pickup(self, obs, me, held, weapons, t) -> Optional[Dict[str, Any]]:
        pos = me["pos"]
        best, best_v = None, -1.0
        for p in obs.get("pickups", []):
            if self.bad_pickups.get(p["id"], 0) > t:
                continue
            if dist2d(p["pos"], pos) > self.reach - 0.3 or abs(p["pos"][2] - pos[2]) > 3.8:
                continue
            if p["kind"] in held:
                continue
            v = WEAPON_VALUE.get(p["kind"], 0)
            if v > best_v:
                best, best_v = p, v
        if best is None:
            return None
        if None in weapons:
            return {"take": best["id"]}
        # both slots full: swap out the weakest weapon if the pickup is better
        worst_i, worst_v = None, 1e9
        for i, w in enumerate(weapons):
            if w is None:
                continue
            if w["kind"] == "laser" and w.get("charge", 0) > 0:
                continue
            v = WEAPON_VALUE.get(w["kind"], 0)
            if v < worst_v:
                worst_i, worst_v = i, v
        if worst_i is not None and best_v > worst_v:
            return {"take": best["id"], "drop": worst_i}
        return None


if __name__ == "__main__":
    mf.run(Bot)
