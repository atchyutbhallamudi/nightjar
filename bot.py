"""Arena bot.

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

import gc
import math
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf

# never let a map or routing problem stop the bot from answering; each degrades on its own
try:
    from world import World
except Exception:
    World = None  # type: ignore
try:
    from nav import NavGrid
except Exception:
    NavGrid = None  # type: ignore

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


# Switches for scoring one change at a time.
FLAGS: Dict[str, Any] = {
    "hunt": True,          # hunt persona while every death so far was our own kill (a squad that never fights itself)
    "lat": 0.075,           # seconds of decision delay added to aim lead
    "bullet_dodge": False, # reverse strafe when a bullet is on course
    "gun_gate": 230.0,     # do not fire the gun at mechs beyond this many meters
    "finish": True,        # close in on a badly hurt target
    "fight_cubes": True,   # step onto nearby cubes while fighting
    "laser_walk": 250.0,   # how far to walk for a laser in hunt persona
    "strafe_boost": 45,    # boost while strafing in a fight when energy is above this (0 = off)
    "fight_weapon": True,  # leave a fight briefly for a laser or lightning within 80 m
    "lethal": True,        # release a partial laser charge when it already kills
    "pos_score": False,    # choose each fight step by scoring nearby spots (range, line of fire, other enemies' sight)
    "ranges": {"gun": 85.0, "laser": 110.0, "lightning": 28.0},  # preferred fight distance per weapon for pos_score
    "slew": 0.05,          # fire when the torso will be on target within this many seconds of turning
    "gun_first": False,    # with gun + laser, aim the gun until the laser is nearly charged
    "spin": 6.0,           # when idle, spend 2 s of every this-many seconds turning the torso a full circle (0 = off)
    "sidestep": 0,         # step out of the path of incoming bullets and laser lines; boost above this energy (0 = off)
    "side_rel": False,     # sidestep judges bullets against our own motion too
    "jet_scan": False,     # at the start, jet up while turning to spot weapons over the rooftops
    "wait_laser": False,   # until we hold a laser and are above 80% HP, fight only attackers and mechs within 55 m
    "plan": False,         # fight movement by 0.8 s look-ahead (range band, line of fire, exposure, bullets, smoothness)
    "fixes": True,         # fire check uses the commanded torso; aim at the visible part; hunters judged by their own approach
    "hold_drop": 0.0,      # stay in a supply ring we are capturing when this many seconds or fewer remain (0 = off)
}


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


def dist2d(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(yaw: float) -> List[float]:
    return [math.cos(yaw), math.sin(yaw)]


class Bot:
    name = "nightjar-v3"

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
        self.nav = None
        if self.world is not None and NavGrid is not None:
            try:
                self.nav = NavGrid(self.world)
            except Exception:
                mf.log("route grid failed, steering locally only:\n" + traceback.format_exc())
        self.path: Optional[List[Tuple[float, float]]] = None
        self.path_goal: Optional[List[float]] = None
        self.path_t = -99.0
        self.search_goal: Optional[List[float]] = None
        self.goal_key: Any = None
        self.goal_best_d = 1e9
        self.goal_best_t = 0.0
        self.bad_goals: Dict[Any, float] = {}
        self.last_flip = -99.0
        self.climbing = False
        self.zone_roof = False
        self.zone_obs: Optional[Dict[str, Any]] = None
        self.force: Optional[str] = None
        self.force_until = 0.0
        self.zprog_best = 1e9
        self.zprog_t = 0.0
        self.zprog_seen = -99.0
        self._safe_cache: Dict[Any, Tuple[List[float], bool]] = {}
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
        # persona: "hunt" until a mech dies that we did not kill, which only happens in a real free-for-all
        self.start_alive: Optional[int] = None
        self.kills = 0
        self.other_dead = 0
        self.ffa = False
        self.hunt = False
        self.lose_t = -99.0
        self.plan_dir: Optional[Tuple[float, float]] = None
        self.plan_t = -99.0
        self.jet_scan = False
        self.stick_t = -99.0
        self.hit_me: Dict[Any, float] = {}
        # remembered loot: kind -> id -> (pos, weapon kind or "", time last seen)
        self.mem: Dict[str, Dict[Any, Tuple[List[float], str, float]]] = {"p": {}, "c": {}, "k": {}, "n": {}}
        # The map and route grid never change: move them out of the garbage collector's sight so its
        # periodic full sweeps stay short (a long pause can blow the 50 ms answer budget).
        gc.collect()
        if hasattr(gc, "freeze"):
            gc.freeze()

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
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act: Dict[str, Any] = {"tick": obs["tick"]}

        self._read_events(obs, t)
        self.zone_obs = obs.get("zone")
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        # persona: a death that was not our kill, seen on 5 decisions in a row, means a real free-for-all
        alive = int(obs.get("alive", 16))
        if self.start_alive is None:
            self.start_alive = alive
        self.other_dead = self.other_dead + 1 if self.start_alive - alive - self.kills > 0 else 0
        if self.other_dead >= 5:
            self.ffa = True
        self.hunt = bool(FLAGS["hunt"]) and not self.ffa

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for e in enemies:
            self.threats[e["id"]] = (t, e["pos"])
        self.threats = {k: v for k, v in self.threats.items() if t - v[0] < 8}
        self._remember(obs, me, npcs, t)

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

        # A supply drop we are already standing in and about to win: full HP, full energy and 3 cubes are worth
        # a few more seconds on the spot, as long as nobody else is in the ring to freeze the clock.
        hold_drop = None
        if FLAGS["hold_drop"]:
            for dr in obs.get("drops", []):
                ring = float(dr.get("radius", 8))
                if float(dr.get("lands_in", 1)) > 0 or dr.get("holder") not in (None, me["id"]):
                    continue
                if dist2d(dr["pos"], me["pos"]) > ring - 0.5:
                    continue
                left = float(dr.get("capture_s", 4)) - float(dr.get("progress", 0))
                crowded = any(dist2d(e["pos"], dr["pos"]) < ring + 3 for e in enemies)
                if left <= float(FLAGS["hold_drop"]) and not crowded and hp_frac > 0.2:
                    hold_drop = dr

        if not me["in_zone"] or zone_urgent:
            mode = "zone"
            self._zone_watchdog(me, zone_goal, t)
            move_yaw = self._nav_yaw(me, zone_goal, t, zone=True)
            if self.zone_roof and dist2d(me["pos"], zone_goal) < 80:
                # the safe spot is a rooftop: head straight for that building and jet up onto it
                move_yaw = math.atan2(zone_goal[1] - me["pos"][1], zone_goal[0] - me["pos"][0])
                self.climbing = True
            boost = (not me["in_zone"]) or me["energy"] > 60
        elif hold_drop is not None:
            mode = "drop"
            dd = dist2d(hold_drop["pos"], me["pos"])
            move_yaw = math.atan2(hold_drop["pos"][1] - me["pos"][1], hold_drop["pos"][0] - me["pos"][0])
            move_mag = 1.0 if dd > 3 else (0.4 if dd > 1 else 0.0)
        elif self.retreating and (target is not None or self.threats):
            # keep hiding from enemies we saw in the last few seconds, even once they drop out of view
            mode = "retreat"
            remembered = [{"pos": p} for (_seen, p) in self.threats.values()]
            move_yaw, move_mag, boost = self._retreat_move(me, near_enemies or remembered, t, zone_goal)
        elif target is not None:
            mode = "fight"
            move_yaw, move_mag, boost = self._fight_move(me, target, tgt_d, tgt_los, held, obs, t)
        else:
            goal, key, loot_target, why = self._loot_goal(obs, me, held, t, zone_goal, hp_frac, npcs)
            if goal is not None:
                mode = why
                d = dist2d(goal, me["pos"])
                self._track_goal(key, d, t)
                move_yaw = self._nav_yaw(me, goal, t)
                move_mag = 1.0 if d > 3 else 0.3
            elif self.last_damage and t - self.last_damage[0] < 3 and self.last_damage[1] is not None:
                mode = "evade"
                move_yaw = self.last_damage[1] + math.pi / 2 * self.strafe_dir
                boost = me["energy"] > 40
                self._maybe_flip_strafe(t)
            else:
                mode = "roam"
                goal = self._wander_goal(obs, me, t, zone_goal)
                move_yaw = self._nav_yaw(me, goal, t)
                # walking slowly is silent (protocol.md section 8)
                move_mag = 0.4 if dist2d(goal, me["pos"]) < 120 else 1.0
                if self.zone_roof and me["in_zone"]:
                    move_mag = 0.0  # made it onto the safe roof: stay put rather than walk off the edge

        # ---------------- get out from under a mortar shell (slow, predictable, 7 m splash)
        away = self._mortar_escape(obs, me)
        if away is not None:
            move_yaw, move_mag, boost = away, 1.0, me["energy"] > 10
            mode = "dodge"
        elif FLAGS["sidestep"] and self.hunt and mode != "zone":
            # step out of the path of bullets that are about to hit, and off the line of a charging laser
            sx, sy, beam = self._sidestep(obs, me)
            if sx or sy:
                vx = (math.cos(move_yaw) * move_mag * 0.4 if move_yaw is not None else 0.0) + sx * 0.9
                vy = (math.sin(move_yaw) * move_mag * 0.4 if move_yaw is not None else 0.0) + sy * 0.9
                move_yaw, move_mag = math.atan2(vy, vx), min(1.0, math.hypot(vx, vy))
                if me["energy"] > (25 if beam else float(FLAGS["sidestep"])):
                    boost = True

        # ---------------- steering
        jump = False
        climb = self.climbing and mode in ("zone", "roam", "weapon", "cube", "drop", "loot")
        if move_yaw is not None and climb and self.world is not None:
            # Jet straight over whatever is in the way: wait (without boosting, which drains energy) until there is
            # enough energy for this wall's height, jump, then jet only while the wall is still ahead.
            px, py, pz = me["pos"]
            top = self.world.obstacle_top(px, py, pz, move_yaw, 6.0)
            if not me["grounded"]:
                boost = top is not None and me["energy"] > 3
            elif top is not None:
                st = me.get("stats", {})
                per_m = float(st.get("jet_drain", 15)) / max(1.0, float(st.get("max_rise_speed", 10)))
                hop = float(st.get("jump_speed", 11)) ** 2 / (2 * float(st.get("gravity", 22)))
                need = min(me["max_energy"] * 0.97, max(0.0, top - pz - hop) * per_m * 1.15 + 8)
                jump = me["energy"] >= need
                move_mag = 1.0 if jump else 0.0
                boost = False
            act["move"] = [round(math.cos(move_yaw) * move_mag, 3), round(math.sin(move_yaw) * move_mag, 3)]
        elif move_yaw is not None:
            move_yaw, jump_needed = self._steer(me, move_yaw, t, dt, move_mag)
            jump = jump or jump_needed
            act["move"] = [round(math.cos(move_yaw) * move_mag, 3), round(math.sin(move_yaw) * move_mag, 3)]
        else:
            act["move"] = [0.0, 0.0]
        if t < self.unstick_until and not me["grounded"] and me["energy"] > 10:
            boost = True  # jet over the obstacle
        if FLAGS["jet_scan"] and self.hunt and "laser" not in held and target is None:
            # opening look-around from the air: sight is not blocked from above the rooftops, so a short jet while
            # the torso turns shows where the weapons lie before the first hunter arrives
            if me["grounded"] and t < 1.5 and me["energy"] >= 90 and not self.jet_scan:
                jump = True
                self.jet_scan = True
            elif self.jet_scan and not me["grounded"] and me["energy"] > 35 and t < 8:
                boost = True
            elif self.jet_scan and (me["grounded"] and t > 2.0 or me["energy"] <= 35 or t >= 8):
                self.jet_scan = False
        else:
            self.jet_scan = False
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
                    if e.get("by") is not None:
                        self.hit_me[e.get("by")] = t
            elif et == "kill":
                self.kills += 1
                self.threats.pop(e.get("target"), None)
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

    def _remember(self, obs: Dict[str, Any], me: Dict[str, Any], npcs: List[Dict[str, Any]], t: float) -> None:
        """Keep loot we have seen after it leaves view; forget it once we look at its spot from close by and it is gone."""
        seen_now = set()
        for tab, items in (("p", obs.get("pickups", [])), ("c", obs.get("cubes", [])), ("k", obs.get("crates", []))):
            for it in items:
                self.mem[tab][it["id"]] = (it["pos"], it.get("kind", ""), t)
                seen_now.add((tab, it["id"]))
        for n in npcs:
            self.mem["n"][n["id"]] = (n["pos"], "", t)
        st = me.get("stats", {})
        eye = [me["pos"][0], me["pos"][1], me["pos"][2] + float(st.get("eye_height", 8))]
        w = self.world
        for tab in ("p", "c", "k"):
            gone = []
            for iid, (pos, _kind, seen) in self.mem[tab].items():
                if (tab, iid) in seen_now or t - seen < 0.35:
                    continue
                if dist2d(pos, me["pos"]) < 40 and (w is None or w.clear_line(eye, [pos[0], pos[1], pos[2] + 1.5])):
                    gone.append(iid)
            for iid in gone:
                del self.mem[tab][iid]
        self.mem["n"] = {k: v for k, v in self.mem["n"].items() if t - v[2] < 120}

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
            self.strafe_until = t + self.rng.uniform(0.6, 1.8)

    def _bullet_incoming(self, obs, me) -> bool:
        """An enemy bullet whose straight path passes within ~4 m of us, arriving in 0.15-1 s.
        Reversing then makes a shooter that led our current velocity miss by several meters."""
        cx, cy = me["pos"][0], me["pos"][1]
        cz = me["pos"][2] + 4.5
        vx, vy, vz = me["vel"]
        for p in obs.get("projectiles", []):
            if p.get("kind") not in ("gun", "npc"):
                continue
            rx, ry, rz = p["pos"][0] - cx, p["pos"][1] - cy, p["pos"][2] - cz
            ux, uy, uz = p["vel"][0] - vx, p["vel"][1] - vy, p["vel"][2] - vz
            uu = ux * ux + uy * uy + uz * uz
            if uu < 1:
                continue
            tca = -(rx * ux + ry * uy + rz * uz) / uu
            if 0.15 < tca < 1.0:
                mx, my, mz = rx + ux * tca, ry + uy * tca, rz + uz * tca
                if math.hypot(mx, my) < 4.0 and abs(mz) < 6:
                    return True
        return False

    def _sidestep(self, obs, me) -> Tuple[float, float, bool]:
        """Unit direction that takes us out of the way of incoming fire, or (0, 0): sideways off the path of each
        bullet that would pass through us within 0.7 s, and sideways off the aim line of a laser being charged at us."""
        st = me.get("stats", {})
        cx, cy = me["pos"][0], me["pos"][1]
        cz = me["pos"][2] + float(st.get("height", 9)) * 0.56
        reach = float(st.get("radius", 2.5)) + 1.1
        dx = dy = 0.0
        for p in obs.get("projectiles", []):
            if p.get("kind") == "mortar":
                continue
            vx, vy, vz = p["vel"]
            # side_rel: judge the bullet against where we will be if we keep moving as we are (a shooter that
            # leads us aims there), not against where we stand now
            ux, uy, uz = vx, vy, vz
            if FLAGS["side_rel"]:
                ux, uy, uz = vx - me["vel"][0], vy - me["vel"][1], vz - me["vel"][2]
            v2 = ux * ux + uy * uy + uz * uz
            if v2 < 1:
                continue
            tc = ((cx - p["pos"][0]) * ux + (cy - p["pos"][1]) * uy + (cz - p["pos"][2]) * uz) / v2
            if tc <= 0 or tc > 0.7:
                continue
            mx, my = p["pos"][0] + ux * tc - cx, p["pos"][1] + uy * tc - cy
            vh = math.hypot(vx, vy)
            if math.hypot(mx, my) < reach and vh >= 1:
                px, py = -vy / vh, vx / vh
                if px * mx + py * my > 0:  # step away from the side the bullet passes on
                    px, py = -px, -py
                dx += px
                dy += py
        beam = False
        for m in obs.get("mechs", []):
            if m.get("kind") != "mech" or not m.get("charging"):
                continue
            to_me = math.atan2(cy - m["pos"][1], cx - m["pos"][0])
            if abs(wrap(m["aim"][0] - to_me)) > 0.15:
                continue
            dx += math.cos(to_me + math.pi / 2 * self.strafe_dir) * 1.5
            dy += math.sin(to_me + math.pi / 2 * self.strafe_dir) * 1.5
            beam = True
        n = math.hypot(dx, dy)
        if n > 0.01:
            return dx / n, dy / n, beam
        return 0.0, 0.0, False

    def _mortar_escape(self, obs, me) -> Optional[float]:
        """Yaw that carries us out of a visible mortar shell's predicted splash, or None."""
        w = self.world
        if w is None:
            return None
        g = self.mortar_g
        splash = float(self.cfg["weapons"]["mortar"].get("splash_radius", 7))
        x, y = me["pos"][0], me["pos"][1]
        best = None
        for p in obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            land = None
            for k in range(1, 100):
                tt = k * 0.05
                lx, ly = px + vx * tt, py + vy * tt
                lz = pz + vz * tt - 0.5 * g * tt * tt
                if lz <= w.ground(lx, ly) + 0.5:
                    land = (lx, ly, tt)
                    break
            if land is None or land[2] > 4.0:
                continue
            d = math.hypot(x - land[0], y - land[1])
            if d < splash + self.mech_r + 4 and (best is None or d < best[0]):
                best = (d, land)
        if best is None:
            return None
        d, (lx, ly, _) = best
        if d < 0.5:
            return math.atan2(obs["projectiles"][0]["vel"][1], obs["projectiles"][0]["vel"][0]) + math.pi / 2
        return math.atan2(y - ly, x - lx)

    def _zone_circle(self, zn: Dict[str, Any], t: float) -> Tuple[float, float, float]:
        """The safe circle a few seconds from now (it slides from the current circle to the next while shrinking)."""
        ss, se = float(zn["shrink_start"]), float(zn["shrink_end"])
        c0, r0 = zn["center"], float(zn["radius"])
        nc, nr = zn["next_center"], float(zn["next_radius"])
        f = 0.0 if se <= ss else min(1.0, max(0.0, (t + 4.0 - ss) / (se - ss)))
        return (c0[0] + (nc[0] - c0[0]) * f, c0[1] + (nc[1] - c0[1]) * f, max(0.0, r0 + (nr - r0) * f - 6.0))

    def _nav_yaw(self, me, goal, t: float, zone: bool = False) -> float:
        """Direction to walk toward `goal`, following an A* route around buildings when needed.
        Routes avoid ground outside the safe circle, and are re-planned often while it shrinks.
        Only when racing the zone (`zone`) may it jet over rooftops instead of a long walk around."""
        x, y = me["pos"][0], me["pos"][1]
        direct = math.atan2(goal[1] - y, goal[0] - x)
        self.climbing = False
        nav = self.nav
        if nav is None:
            return direct
        if zone and self.force == "climb" and t < self.force_until:
            self.climbing = True  # watchdog: walking made no progress, go straight over
            return direct
        d = math.hypot(goal[0] - x, goal[1] - y)
        # close in (goal may sit right against a wall, inside a grid cell marked blocked) or in the open: go direct
        if d < 15 or (d < 90 and nav.line_free(x, y, goal[0], goal[1])):
            self.path = None
            return direct
        new_goal = self.path_goal is None or math.hypot(goal[0] - self.path_goal[0], goal[1] - self.path_goal[1]) > 6
        zn = self.zone_obs
        shrinking = zn is not None and float(zn["shrink_start"]) - 5 <= t <= float(zn["shrink_end"])
        stale = t - self.path_t > (3.0 if shrinking else 12.0)
        # self.path: None = no route yet, [] = no route exists (retry only once stale)
        if new_goal or stale or self.path is None:
            same_search = nav.searching() and self.search_goal is not None and \
                math.hypot(goal[0] - self.search_goal[0], goal[1] - self.search_goal[1]) < 6
            if not same_search:
                keep = self._zone_circle(zn, t) if zn is not None else None
                if not nav.start(x, y, goal[0], goal[1], keep=keep):
                    self.path, self.path_goal, self.path_t = [], [goal[0], goal[1]], t
                    return direct
                self.search_goal = [goal[0], goal[1]]
        if nav.searching():
            res = nav.step(0.010)
            if res is not None:
                path = nav.smooth(res) if res else []
                if path:
                    walk = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                               for i in range(len(path) - 1))
                    straight = math.hypot(goal[0] - x, goal[1] - y)
                    forced_walk = self.force == "path" and t < self.force_until
                    if zone and straight < 160 and walk > 1.8 * straight + 30 and not forced_walk and \
                            self.world is not None and \
                            self.world.max_top_along(x, y, goal[0], goal[1]) < me["pos"][2] + 50:
                        path = []  # the walk goes far around: go straight over the (climbable) rooftops instead
                self.path = path
                self.path_goal = list(self.search_goal or goal)
                self.path_t = t
        if not self.path or self.path_goal is None or \
                math.hypot(goal[0] - self.path_goal[0], goal[1] - self.path_goal[1]) > 6:
            # no walkable route (e.g. inside the stadium bowl): head straight there and jet over walls
            self.climbing = self.path == [] and not nav.searching()
            return direct
        path = self.path
        while len(path) > 1 and math.hypot(path[0][0] - x, path[0][1] - y) < 5:
            path.pop(0)
        tgt = path[0]
        for wp in path[1:6]:
            if nav.line_free(x, y, wp[0], wp[1]):
                tgt = wp
            else:
                break
        if not nav.line_free(x, y, tgt[0], tgt[1]) and math.hypot(tgt[0] - x, tgt[1] - y) > 12:
            self.path_t = -99.0  # knocked off the route: plan again next decision
        return math.atan2(tgt[1] - y, tgt[0] - x)

    def _zone_watchdog(self, me, goal, t: float) -> None:
        """Heading for the zone but not getting closer for 4 s: switch between walking a route and
        climbing straight over, for 8 s at a time."""
        d = dist2d(me["pos"], goal)
        if t - self.zprog_seen > 1.0:  # (re)entering zone mode: start fresh
            self.zprog_best, self.zprog_t = d, t
        self.zprog_seen = t
        if d < self.zprog_best - 2:
            self.zprog_best, self.zprog_t = d, t
        elif t - self.zprog_t > 4 and d > 10:
            self.force = "path" if self.force == "climb" else "climb"
            self.force_until = t + 8
            self.zprog_best, self.zprog_t = d, t
            self.path_t = -99.0  # plan again under the new strategy

    def _track_goal(self, key, d: float, t: float) -> None:
        """Give up on a loot goal we have not got 3 m closer to in 7 s (unreachable, e.g. on a roof)."""
        if key != self.goal_key:
            self.goal_key, self.goal_best_d, self.goal_best_t = key, d, t
            return
        if d < self.goal_best_d - 3:
            self.goal_best_d, self.goal_best_t = d, t
        elif t - self.goal_best_t > 10 and d > 4:
            self.bad_goals[key] = t + 30
            self.goal_key = None

    # ------------------------------------------------------------------ targets and modes

    def _choose_target(self, me: Dict[str, Any], enemies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Lowest score wins: near, hurt, in the open, and (rules edges) showing us its back
        (rear hits do x1.5), slowed by charging a laser, or carrying a mortar when we hold a laser (cook-off +60%)."""
        if self.hunt:
            return self._choose_target_hunt(me, enemies)
        best, best_s = None, 1e9
        my_weapons = {w["kind"] for w in me["weapons"] if w}
        for e in enemies:
            d = dist2d(e["pos"], me["pos"])
            hpf = float(e.get("hp", 1)) / max(1.0, float(e.get("max_hp", 1) or 1))
            s = d + 70 * hpf
            if float(e.get("hp", 9999)) < 200:
                s -= 50  # finish it: a kill is a placement
            if not self._los(me, e):
                s += 120
            to_me = math.atan2(me["pos"][1] - e["pos"][1], me["pos"][0] - e["pos"][0])
            if abs(wrap(e["aim"][0] - to_me)) > math.pi / 2:
                s -= 30
            if e.get("charging"):
                s -= 10
            if "laser" in my_weapons and "mortar" in (e.get("weapons") or []):
                s -= 20
            if e["id"] == self.target_id:
                s -= 35
            if s < best_s:
                best, best_s = e, s
        self.target_id = best["id"] if best else None
        return best

    def _choose_target_hunt(self, me: Dict[str, Any], enemies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Hunt persona: fight the mechs that are coming for us (near, closing, aiming at us, or that hit us)
        and leave idle ones alone. Stay on one target, and strongly prefer the one closest to dying."""
        t = self.prev_t
        zn = self.zone_obs or {}
        late = float(zn.get("next_radius", 999)) <= 70
        mx, my = me["pos"][0], me["pos"][1]
        best, best_s = None, 1e9
        hidden, hidden_d = None, 1e9
        for e in enemies:
            dx, dy = e["pos"][0] - mx, e["pos"][1] - my
            d = math.hypot(dx, dy)
            if d > 240:
                continue
            hp = float(e.get("hp", 1400))
            mhp = max(1.0, float(e.get("max_hp", 1400) or 1400))
            closing = 0.0
            if d > 1:
                closing = -((e["vel"][0] - me["vel"][0]) * dx + (e["vel"][1] - me["vel"][1]) * dy) / d
                if FLAGS["fixes"]:
                    # count only the enemy's own approach: our walking toward a mech that is standing still
                    # does not make it a hunter
                    closing = -(e["vel"][0] * dx + e["vel"][1] * dy) / d
            aimed = abs(wrap(e["aim"][0] - math.atan2(-dy, -dx))) < 0.22
            hit = t - self.hit_me.get(e["id"], -99.0)
            engage = d < 60 or hit < 6 or late or (closing > 3 and d < 180) or (hp < 0.4 * mhp and d < 160) or \
                (aimed and d < 190) or e["id"] == self.target_id and t - self.stick_t < 2
            if FLAGS["wait_laser"] and not late:
                # start nothing until we hold a laser and are healthy: fight only what has hit us or is on top of us
                armed = any(w and w["kind"] == "laser" for w in me["weapons"]) and me["hp"] >= 0.8 * me["max_hp"]
                if not armed:
                    engage = hit < 10 or d < 55
            if not engage:
                continue
            if not self._los(me, e):
                if d < hidden_d:
                    hidden, hidden_d = e, d
                continue
            s = d + 0.18 * hp
            if hp <= 0.3 * mhp:
                s -= 200
            if e["id"] == self.target_id and t - self.stick_t < 2:
                s -= 200
            if hit < 4:
                s -= 40
            if closing > 3:
                s -= 15
            if s < best_s:
                best, best_s = e, s
        if best is None:
            best = hidden  # no clear shot at any of them: close in on the nearest
        if best is not None:
            self.target_id, self.stick_t = best["id"], t
        return best

    def _safe_point(self, nc: List[float], nr: float) -> Tuple[List[float], bool]:
        """A standable spot well inside the next circle: the open ground cell nearest its center.
        If none lies inside (center in a big building), return the center and ask to climb onto the roof."""
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        hit = self._safe_cache.get(key)
        if hit is not None:
            return hit
        res: Tuple[List[float], bool] = ([nc[0], nc[1]], False)
        nav = self.nav
        if nav is not None:
            ci, cj = nav.to_cell(nc[0], nc[1])
            r_cells = max(0, min(20, int(nr * 0.6 / nav.cell)))
            f = nav.nearest_free(ci, cj, max_r=r_cells)
            if f is not None:
                cx, cy = nav.center(*f)
                res = ([cx, cy], False)
            else:
                res = ([nc[0], nc[1]], nr > 3)
        self._safe_cache[key] = res
        return res

    def _zone_goal(self, obs: Dict[str, Any], me: Dict[str, Any], t: float) -> Tuple[List[float], bool]:
        zn = obs["zone"]
        nc, nr = zn["next_center"], float(zn["next_radius"])
        goal, self.zone_roof = self._safe_point(nc, nr)
        d = dist2d(me["pos"], nc)
        margin = min(25.0, nr * 0.25)
        need = d - (nr - margin)
        if need <= 0:
            return goal, False
        # The boundary starts moving at shrink_start, sliding from the current circle to the next one.
        # Find when it would pass over where we stand now, and leave in time (walking, ~30% detour).
        speed = float(me.get("stats", {}).get("walk_speed", 11.7))
        ss, se = float(zn["shrink_start"]), float(zn["shrink_end"])
        c0, r0 = zn["center"], float(zn["radius"])
        reach = se
        for k in range(21):
            f = k / 20
            cx, cy = c0[0] + (nc[0] - c0[0]) * f, c0[1] + (nc[1] - c0[1]) * f
            if math.hypot(me["pos"][0] - cx, me["pos"][1] - cy) > r0 + (nr - r0) * f:
                reach = ss + (se - ss) * f
                break
        urgent = need * 1.3 / speed + 10 > reach - t
        return goal, urgent

    def _update_retreat(self, hp_frac: float, near: List[Dict[str, Any]], target, tgt_d: float, t: float) -> None:
        if self.hunt:
            # Hunt persona: points are kills, so only step back while clearly losing, and only briefly.
            n = len(near)
            losing = (n >= 2 and hp_frac < 0.55) or (n >= 3 and hp_frac < 0.8)
            if target is not None:
                tf = float(target.get("hp", 1400)) / max(1.0, float(target.get("max_hp", 1400) or 1400))
                if hp_frac < 0.35 and tf > hp_frac + 0.1:
                    losing = True
                if tf < 0.25 and hp_frac > 0.2 and n <= 2:
                    losing = False
            if losing:
                if not self.retreating:
                    self.cover = None
                self.lose_t = t
            self.retreating = t - self.lose_t < 1.5
            return
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

    def _fight_move(self, me, target, d, los, held, obs, t) -> Tuple[float, float, bool]:
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
        finishing = False
        if self.hunt and FLAGS["finish"]:
            thp = float(target.get("hp", 1400))
            tmax = max(1.0, float(target.get("max_hp", 1400) or 1400))
            if thp < 0.38 * tmax and me["hp"] > 0.42 * me["max_hp"] and len(self.threats) <= 2:
                lo, hi = 10.0, 20.0  # badly hurt target: run it down before it can heal
                finishing = True
        if self.hunt:
            # hold each strafe direction for a few seconds; quick reversals keep us close to standing still
            if t >= self.strafe_until:
                self.strafe_dir = -self.strafe_dir
                self.strafe_until = t + self.rng.uniform(3.0, 5.0)
            if FLAGS["bullet_dodge"] and t - self.last_flip > 0.5 and self._bullet_incoming(obs, me):
                self.strafe_dir = -self.strafe_dir
                self.last_flip = t
        else:
            self._maybe_flip_strafe(t)
        if self.hunt and FLAGS["fight_cubes"]:
            # a kill drops the victim's cubes plus two; each one heals and adds damage, so step onto them mid-fight
            reach = 70.0 if me["hp"] > 0.5 * me["max_hp"] else 40.0
            cube = None
            for pos, _kind, _seen in self.mem["c"].values():
                cd = dist2d(pos, me["pos"])
                if cd < reach and abs(pos[2] - me["pos"][2]) < 4:
                    reach, cube = cd, pos
            if cube is not None:
                return self._nav_yaw(me, cube, t), 1.0, energy > 50
        if self.hunt and FLAGS["fight_weapon"] and "laser" not in held:
            # a better weapon lying close by is worth a few seconds of the fight
            wp, wd = None, 80.0
            for pid, (pos, kind, _seen) in self.mem["p"].items():
                if kind == "mortar" or kind in held or self.bad_pickups.get(pid, 0) > t:
                    continue
                if kind != "laser" and len(held) >= 2:
                    continue  # both slots full: only a laser is worth the swap
                pd = dist2d(pos, me["pos"])
                if pd < wd and abs(pos[2] - me["pos"][2]) < 4:
                    wp, wd = pos, pd
            if wp is not None:
                return self._nav_yaw(me, wp, t), 1.0, energy > 50
        if self.hunt and FLAGS["plan"] and self.world is not None and self.nav is not None:
            # decide the speed first, so the look-ahead weighs the move we will really make
            fast_all = finishing and d > 40 and energy > 30
            if FLAGS["strafe_boost"] and los:
                li = held.get("laser")
                charging = li is not None and me["weapons"][li].get("charge", 0) > 0
                if not charging and energy > (60 if "lightning" in held else float(FLAGS["strafe_boost"])):
                    fast_all = True
            plan = self._fight_plan(me, target, d, los, held, obs, t, 17.0 if finishing else None, fast_all)
            if plan is not None:
                return plan
        if self.hunt and FLAGS["pos_score"] and self.world is not None and self.nav is not None:
            spot = self._fight_spot(me, target, d, held, obs, t, 17.0 if finishing else None)
            if spot is not None:
                yaw, mag = spot
                boost = finishing and d > 40 and energy > 30
                if FLAGS["strafe_boost"] and mag > 0:
                    li = held.get("laser")
                    charging = li is not None and me["weapons"][li].get("charge", 0) > 0
                    if not charging and energy > (60 if "lightning" in held else float(FLAGS["strafe_boost"])):
                        boost = True
                return yaw, mag, boost
        radial = 0.0
        if d > hi:
            radial = 1.0
        elif d < lo:
            radial = -1.0
        side = to_t + math.pi / 2 * self.strafe_dir
        vx = radial * math.cos(to_t) + math.cos(side) * (0.8 if radial else 1.0)
        vy = radial * math.sin(to_t) + math.sin(side) * (0.8 if radial else 1.0)
        boost = (radial > 0 and d > hi + 30 and energy > 55) or (radial < 0 and energy > 35)
        if finishing and d > 40 and energy > 30:
            boost = True
        if self.hunt and FLAGS["strafe_boost"] and los:
            # a faster sideways target is harder to lead; keep energy back for lightning and never boost a charge
            li = held.get("laser")
            charging = li is not None and me["weapons"][li].get("charge", 0) > 0
            if not charging and energy > (60 if "lightning" in held else float(FLAGS["strafe_boost"])):
                boost = True
        return math.atan2(vy, vx), 1.0, boost

    def _fight_plan(self, me, target, d, los_now, held, obs, t, want_range: Optional[float],
                    fast_all: bool = False) -> Optional[Tuple[float, float, bool]]:
        """Choose the next 0.8 s of fight movement by looking ahead. Each candidate (stand, 16 walking directions,
        8 boosting directions when under fire) is judged by where it leaves us: inside the weapon's range band,
        with a line of fire on the target, out of the sights of other mechs in proportion to how hard they could
        hit us from there, clear of bullets already in the air (measured against the candidate's own motion),
        sideways to the target, and not a sharp change from how we are already moving.
        Returns (yaw, speed 0..1, boost) or None when we are not on the ground grid."""
        w, nav = self.world, self.nav
        x, y, z = me["pos"]
        if abs(z - w.ground(x, y)) > 3.0:
            return None
        if not nav.free(*nav.to_cell(x, y)):
            return None  # hard against a wall: the grid calls every step from here blocked, so let the plain strafe steer
        st = me.get("stats", {})
        walk = float(st.get("walk_speed", 11.7))
        fast = float(st.get("boost_speed", 19.5))
        mh = float(st.get("muzzle_height", 7))
        body_h = float(st.get("height", 9)) * 0.6
        my_r = float(st.get("radius", 2.5))
        horizon = 0.8
        rng = FLAGS["ranges"]
        pref = rng["gun"]
        if "lightning" in held and me["energy"] > 12:
            pref = rng["lightning"]
        elif "laser" in held:
            pref = rng["laser"]
        tw = target.get("weapons") or []
        if "lightning" not in held:
            if "lightning" in tw:
                pref = max(pref, 64.0)
            if "mortar" in tw:
                pref = max(pref, 80.0)
        if want_range is not None:
            pref = want_range
        band = 8.0 if want_range is None else 3.0
        _tr, th = self._size(target)
        tp = target["pos"]
        tc = [tp[0], tp[1], tp[2] + th * 0.5]
        to_t = math.atan2(tp[1] - y, tp[0] - x)
        sx, sy = math.cos(to_t + math.pi / 2 * self.strafe_dir), math.sin(to_t + math.pi / 2 * self.strafe_dir)
        # everyone who could shoot us: (eye, pos, aim yaw, base damage rate, charging a laser, weight)
        threats = []
        for e in obs["mechs"]:
            if e["kind"] != "mech" or e["id"] == me["id"]:
                continue
            ed = dist2d(e["pos"], me["pos"])
            if ed > 230:
                continue
            _er, eh = self._size(e)
            ew = e.get("weapons") or []
            rate = 120.0 if "laser" in ew else 150.0
            if "lightning" in ew and ed < 50:
                rate = 200.0
            weight = 0.2 if e["id"] == target["id"] else 1.0  # we choose to stand in our own target's sights
            threats.append((ed, [e["pos"][0], e["pos"][1], e["pos"][2] + eh * 8.0 / 9.0], e["pos"], e["aim"][0], rate,
                            bool(e.get("charging")), weight))
        threats.sort(key=lambda q: q[0])
        threats = threats[:3]  # each one costs a sight line per candidate
        bullets = [p for p in obs.get("projectiles", []) if p.get("kind") != "mortar"]
        hurt = t - self.last_hurt_t < 1.5
        under_fire = hurt or bool(bullets) or any(q[5] for q in threats)
        reserve = 45.0 if "lightning" in held else 25.0
        can_boost = under_fire and me["energy"] > reserve and me.get("grounded", True)
        zn = self.zone_obs
        zc = self._zone_circle(zn, t) if zn else None
        cube, cube_d = None, 40.0
        for pos, _k, _s in self.mem["c"].values():
            cd = dist2d(pos, me["pos"])
            if cd < cube_d and abs(pos[2] - z) < 4:
                cube, cube_d = pos, cd
        vx0, vy0 = me["vel"][0], me["vel"][1]
        sp0 = math.hypot(vx0, vy0)
        aimed = abs(wrap(target["aim"][0] - (to_t + math.pi))) < 0.35 and d >= 30 and los_now
        cz_now = z + float(st.get("height", 9)) * 0.56
        best_s, best = -1e18, (to_t, 0.0, False)
        best_dir = None
        n_ok = 0
        side_blocked = False
        n = 24 if (can_boost and not fast_all) else 16  # fast_all: every move is a boosted one, so weigh them as such
        t_start = time.perf_counter()
        for k in range(-1, n):
            if k >= 8 and time.perf_counter() - t_start > 0.020:
                break  # safety net for an overloaded machine: decide with what has been weighed so far
            if k < 0:
                dx = dy = spd = 0.0
            elif k < 16:
                a = k * math.pi / 8
                dx, dy, spd = math.cos(a), math.sin(a), (fast if fast_all else walk)
            else:
                a = (k - 16) * math.pi / 4 + math.pi / 8
                dx, dy, spd = math.cos(a), math.sin(a), fast
            r = spd * horizon * 0.85
            cx, cy = x + dx * r, y + dy * r
            if k >= 0 and not nav.line_free(x, y, cx, cy):
                if k < 16 and dx * sx + dy * sy > 0.9:
                    side_blocked = True
                continue
            cz = w.ground(cx, cy)
            s = -1.5 * max(0.0, abs(math.hypot(tp[0] - cx, tp[1] - cy) - pref) - band)
            los = w.clear_line([cx, cy, cz + mh], tc)
            if los:
                s += 30.0
            body = [cx, cy, cz + body_h]
            for _ed, eye, epos, eaim, rate, charging, weight in threats:
                td = math.hypot(epos[0] - cx, epos[1] - cy)
                if td > 230 or not w.clear_line(eye, body):
                    continue
                off = abs(wrap(eaim - math.atan2(cy - epos[1], cx - epos[0])))
                align = max(0.0, 1.0 - off / 0.7)
                s -= weight * 0.5 * rate * (0.4 + 0.6 * align) * horizon * (1.0 if td < 150 else 0.5)
                if charging and align > 0.3:
                    s -= 0.8 * 252.0 * (weight if weight == 1.0 else 0.6)  # be out of sight when the beam is let go
            cvx, cvy = dx * spd, dy * spd
            for p in bullets:
                ux, uy, uz = p["vel"][0] - cvx, p["vel"][1] - cvy, p["vel"][2]
                u2 = ux * ux + uy * uy + uz * uz
                if u2 < 1:
                    continue
                tca = ((x - p["pos"][0]) * ux + (y - p["pos"][1]) * uy + (cz_now - p["pos"][2]) * uz) / u2
                if 0 < tca <= horizon:
                    miss = math.hypot(p["pos"][0] + ux * tca - x, p["pos"][1] + uy * tca - y)
                    if miss < my_r + 1.5:
                        s -= 1.2 * 45.0 * (1.0 - miss / (my_r + 1.5))
            if zc is not None:
                out = math.hypot(cx - zc[0], cy - zc[1]) - zc[2]
                if out > -5:
                    s -= 150 + 5 * (out + 5)
            if cube is not None:
                s += 1.5 * (cube_d - math.hypot(cube[0] - cx, cube[1] - cy))
            if k >= 0:
                s += (22.0 if aimed else 9.0) * (dx * sx + dy * sy)
                if sp0 > 1:
                    s += 3.0 * (dx * vx0 + dy * vy0) / sp0
                if k >= 16:
                    s -= 3.4
            else:
                s -= 1.0 + (22.0 if hurt else 0.0)
            if k >= 0 and self.plan_dir is not None and t - self.plan_t < 0.35:
                # stay with the heading we chose last time unless another is clearly better: a mover that keeps
                # changing its mind ends up nearly standing still
                s += 14.0 * max(0.0, dx * self.plan_dir[0] + dy * self.plan_dir[1])
            if s > best_s:
                best_s, best = s, (math.atan2(dy, dx) if k >= 0 else to_t, 1.0 if k >= 0 else 0.0,
                                   k >= 16 or (fast_all and k >= 0))
                best_dir = (dx, dy) if k >= 0 else None
                n_ok += k >= 0
            elif k >= 0:
                n_ok += 1
        if n_ok < 4:
            return None  # nearly boxed in: no real choice to weigh, use the plain strafe and its wall steering
        self.plan_dir, self.plan_t = best_dir, t
        if side_blocked and t - self.last_flip > 2.0:
            # the way we were strafing is walled off: turn round once and keep going that way
            self.strafe_dir = -self.strafe_dir
            self.strafe_until = t + self.rng.uniform(3.0, 5.0)
            self.last_flip = t
        return best

    def _fight_spot(self, me, target, d, held, obs, t, want_range: Optional[float]) -> Optional[Tuple[float, float]]:
        """Pick where to stand for the next moment of a fight: score the spot we are on and two rings of nearby
        walkable spots, and head for the best. A good spot is at our weapon's range from the target, has a line
        of fire while we can shoot (and none while the gun reloads), is out of sight of every other enemy, is
        inside the zone, and lies in the direction we are strafing. Returns (yaw, speed 0..1) or None."""
        w, nav = self.world, self.nav
        x, y, z = me["pos"]
        if abs(z - w.ground(x, y)) > 3.0:
            return None  # on a roof or in the air: the ground grid does not describe where we are
        st = me.get("stats", {})
        mh = float(st.get("muzzle_height", 7))
        body_h = float(st.get("height", 9)) * 0.6
        weapons = me["weapons"]
        gun = weapons[held["gun"]] if "gun" in held else None
        laser = weapons[held["laser"]] if "laser" in held else None
        light = weapons[held["lightning"]] if "lightning" in held else None
        rng = dict(FLAGS["ranges"])
        pref = rng["gun"]
        if light is not None and me["energy"] > 12:
            pref = rng["lightning"]
        elif laser is not None:
            pref = rng["laser"]
        if light is None:
            tw = target.get("weapons") or []
            if "lightning" in tw:
                pref = max(pref, 64.0)
            if "mortar" in tw:
                pref = max(pref, 80.0)
        if want_range is not None:
            pref = want_range
        # can we shoot right now? with only a gun, step out of the line of fire while it reloads
        ready = light is not None or laser is not None
        if not ready and gun is not None:
            ready = gun.get("ammo", 1) > 0 and gun.get("reload", 0) <= 0.25 and d <= gun["stats"].get("range", 230) - 10
        want_los = 1.0 if ready else -1.0
        _r, th = self._size(target)
        tp = target["pos"]
        tc = [tp[0], tp[1], tp[2] + th * 0.5]
        others = []
        for e in obs["mechs"]:
            if e["kind"] == "mech" and e["id"] != target["id"] and e["id"] != me["id"]:
                _er, eh = self._size(e)
                others.append((dist2d(e["pos"], me["pos"]), [e["pos"][0], e["pos"][1], e["pos"][2] + eh * 8.0 / 9.0], e["pos"]))
        others.sort(key=lambda o: o[0])
        others = others[:3]
        zn = self.zone_obs
        zc = self._zone_circle(zn, t) if zn else None
        cube = None
        cube_d = 40.0
        for pos, _k, _s in self.mem["c"].values():
            cd = dist2d(pos, me["pos"])
            if cd < cube_d and abs(pos[2] - z) < 4:
                cube, cube_d = pos, cd
        to_t = math.atan2(tp[1] - y, tp[0] - x)
        sx, sy = math.cos(to_t + math.pi / 2 * self.strafe_dir), math.sin(to_t + math.pi / 2 * self.strafe_dir)
        best_s, best = -1e18, (0.0, 0.0, 0.0)
        side_blocked = False
        for k in range(-1, 32):
            if k < 0:
                cx, cy, dx, dy, r = x, y, 0.0, 0.0, 0.0
            else:
                a = (k % 16) * math.pi / 8
                r = 6.0 * (1 + k // 16)
                dx, dy = math.cos(a), math.sin(a)
                cx, cy = x + dx * r, y + dy * r
                if not nav.line_free(x, y, cx, cy):
                    if k < 16 and dx * sx + dy * sy > 0.9:
                        side_blocked = True
                    continue
            cz = w.ground(cx, cy)
            s = -1.1 * abs(math.hypot(tp[0] - cx, tp[1] - cy) - pref) - 0.4 * r
            los = w.clear_line([cx, cy, cz + mh], tc)
            if los:
                s += 45.0 * want_los
            body = [cx, cy, cz + body_h]
            for _od, eye, opos in others:
                if math.hypot(opos[0] - cx, opos[1] - cy) < 170 and w.clear_line(eye, body):
                    s -= 30.0
            if zc is not None:
                out = math.hypot(cx - zc[0], cy - zc[1]) - zc[2]
                if out > -5:
                    s -= 150 + 5 * (out + 5)
            if cube is not None:
                s += 1.5 * (cube_d - math.hypot(cube[0] - cx, cube[1] - cy))
            if want_los > 0:
                s += 9.0 * (dx * sx + dy * sy)
            if k < 0:
                s -= 8.0
                if want_los < 0 and not los:
                    s += 14.0
            if s > best_s:
                best_s, best = s, (dx, dy, r)
        if side_blocked:
            self.strafe_dir = -self.strafe_dir
            self.strafe_until = t + 0.5
        if best[2] == 0.0:
            return to_t, 0.0
        return math.atan2(best[1], best[0]), 1.0

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
        best, best_c, why, key = None, 1e9, "loot", None
        bad = self.bad_goals
        w = self.world

        def reachable(p: List[float]) -> bool:
            # on the ground anywhere (hills included), or on the same roof/level as us
            return abs(p[2] - z) < 3.5 or (w is not None and abs(p[2] - w.ground(p[0], p[1])) < 1.5)

        hunt = self.hunt
        if hunt:  # hunt persona also walks to loot it remembers, not only what is in view right now
            picks = [{"id": i, "kind": k, "pos": p} for i, (p, k, _s) in self.mem["p"].items()]
            cubes = [{"id": i, "pos": p} for i, (p, _k, _s) in self.mem["c"].items()]
        else:
            picks, cubes = obs.get("pickups", []), obs.get("cubes", [])
        for p in picks:
            if self.bad_pickups.get(p["id"], 0) > t or bad.get(("p", p["id"]), 0) > t or not reachable(p["pos"]):
                continue
            if p["kind"] in held or WEAPON_VALUE.get(p["kind"], 0) <= worst:
                continue
            if hunt and p["kind"] == "mortar":
                continue  # a mortar carrier takes +60% from lasers, and the shell rarely lands
            c = dist2d(p["pos"], pos) - 40 * (WEAPON_VALUE[p["kind"]] - worst)
            limit = 220 if ("p", p["id"]) == self.goal_key else 150  # a route may lead away at first
            if hunt and p["kind"] == "laser":
                limit = max(limit, float(FLAGS["laser_walk"]))
                c -= 60
            if dist2d(p["pos"], pos) < limit and c < best_c:
                best, best_c, why, key = p["pos"], c, "weapon", ("p", p["id"])
        for c in cubes:
            d = dist2d(c["pos"], pos)
            if d < 90 and reachable(c["pos"]) and d - 20 < best_c and bad.get(("c", c["id"]), 0) <= t:
                best, best_c, why, key = c["pos"], d - 20, "cube", ("c", c["id"])
        for dr in obs.get("drops", []):
            d = dist2d(dr["pos"], pos)
            speed = float(me.get("stats", {}).get("walk_speed", 11.7))
            if d < 220 and float(dr.get("lands_in", 0)) < d / speed + 8 and dr.get("holder") in (None, me["id"]) \
                    and bad.get(("d", dr["id"]), 0) <= t:
                c = d - 60
                if c < best_c:
                    best, best_c, why, key = dr["pos"], c, "drop", ("d", dr["id"])
        # Something to shoot: a crate (cube + maybe a weapon) or, early on, a drone (2 cubes).
        # Healing needs 5 s without dealing damage, so only start on these at (nearly) full HP,
        # except a drone that is already shooting at us.
        shoot = None
        sd = 1e9
        healthy = hp_frac >= (0.5 if hunt else 0.95)
        npc_on_us = self.last_damage is not None and t - self.last_damage[0] < 3 and self.last_damage[3] == "npc"
        if hunt and best is None and healthy:
            # nothing to pick up: walk toward a remembered crate until it is in view to shoot
            for cid, (cp, _k, _s) in self.mem["k"].items():
                d = dist2d(cp, pos)
                if 30 < d < 150 and d < best_c and reachable(cp) and bad.get(("k", cid), 0) <= t:
                    best, best_c, why, key = cp, d, "crate", ("k", cid)
        if healthy:
            for cr in obs.get("crates", []):
                d = dist2d(cr["pos"], pos)
                if d < 130 and d < sd and self._clear_to(me, [cr["pos"][0], cr["pos"][1], cr["pos"][2] + 0.8]):
                    shoot, sd = {"kind": "crate", "pos": cr["pos"], "vel": [0, 0, 0], "id": cr["id"]}, d
        if (healthy and (t < 240 or hunt) and (hp_frac > 0.6 or not hunt)) or (npc_on_us and hp_frac > 0.35):
            for n in npcs:
                d = dist2d(n["pos"], pos)
                if d < 120 and d < sd and self._los(me, n):
                    shoot, sd = n, d
        return best, key, shoot, why

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
        c_clear = self._clear_to(me, [tgt[0], tgt[1], center_z])
        u_clear = (not c_clear) and obj.get("kind") != "crate" and \
            self._clear_to(me, [tgt[0], tgt[1], tgt[2] + height * 0.85])
        los = c_clear or u_clear
        if u_clear and FLAGS["fixes"]:
            # only the upper body shows over cover: aim at the part we can see, not at the wall in front of its middle
            center_z = tgt[2] + height * 0.85
            d3 = math.dist(muzzle, [tgt[0], tgt[1], center_z])

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
        laser_ok = bool(laser and los and d3 <= laser["stats"].get("range", 280))
        # gun_first: keep the gun aimed and firing through the laser's cooldown and the first part of its charge,
        # and only swing to the beam's straight line once the charge is nearly worth releasing
        gun_first = bool(FLAGS["gun_first"]) and self.hunt and is_mech and gun is not None and \
            gun.get("ammo", 0) > 0 and not gun.get("reload")
        if gun_first and laser_ok:
            if laser.get("charge", 0) > 0.45 * laser["stats"].get("charge", 0.75):
                primary = "laser"
        elif laser_ok and (laser.get("charge", 0) > 0 or (laser["cooldown"] < 0.3 and not retreating)):
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

        mst = me.get("stats", {})
        slew = float(FLAGS["slew"])  # seconds of torso turn we count on before the shot leaves
        slew_y = float(mst.get("turn_rate", 4)) * slew
        slew_p = float(mst.get("pitch_rate", 3)) * slew

        # Both weapons leave along the one torso. Where it will point when the shot goes: its heading now, turned
        # toward the aim we are commanding this decision (not toward whatever each weapon would have liked).
        turn_y = max(-slew_y, min(slew_y, wrap(aim[0] - my_aim[0])))
        turn_p = max(-slew_p, min(slew_p, aim[1] - my_aim[1]))
        torso = [my_aim[0] + turn_y, my_aim[1] + turn_p]

        def on_target(want: List[float], tol: float) -> bool:
            if FLAGS["fixes"]:
                ey = abs(wrap(want[0] - torso[0]))
                ep = abs(want[1] - torso[1])
            else:
                ey = max(0.0, abs(wrap(want[0] - my_aim[0])) - slew_y)
                ep = max(0.0, abs(want[1] - my_aim[1]) - slew_p)
            return d3 * math.hypot(ey * math.cos(want[1]), ep) < tol

        # gun
        if gun_i is not None and gun and los and d3 <= gun["stats"].get("range", 230) and \
                (not is_mech or d3 <= float(FLAGS["gun_gate"])):
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
                if retreating and charge > 0:
                    # don't keep walking at half speed for a full charge: fire what we have once on target
                    fire[laser_i] = not on_target(aim, radius * 0.8)
                elif FLAGS["lethal"] and self.hunt and is_mech and charge >= laser["stats"].get("min_charge", 0.15) \
                        and charge < full - 0.02 and self._laser_now(me, laser, charge, full, obj) * 0.7 >= \
                        float(obj.get("hp", 1e9)) and on_target(aim, radius * 0.8):
                    fire[laser_i] = False  # what is charged already kills: let go now
                elif laser["cooldown"] <= 0.05 and charge < full - 0.02:
                    fire[laser_i] = True
                elif charge > 0:
                    fire[laser_i] = not on_target(aim, radius * 0.7)
            elif gun_first and laser_ok and not retreating and laser["cooldown"] <= 0.05 and charge < full - 0.02:
                fire[laser_i] = True  # charge the beam while the gun does the aiming
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

    def _laser_now(self, me, laser, charge: float, full: float, obj) -> float:
        """Damage a laser shot would do if released at this charge (cube bonus and mortar cook-off included)."""
        st = laser["stats"]
        mf = float(st.get("min_frac", 0.35))
        dmg = float(st.get("damage", 315)) * (mf + (1 - mf) * min(1.0, charge / max(0.01, full)))
        cc = self.cfg.get("cubes", {})
        dmg *= 1 + float(cc.get("damage_per_cube", 0.04)) * min(float(me.get("cubes", 0)), float(cc.get("stat_cap", 10)))
        if "mortar" in (obj.get("weapons") or []):
            dmg *= 1 + float(st.get("cookoff", 0.6))
        return dmg

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
            lat = float(FLAGS["lat"])
            for _ in range(3):
                tt = math.dist(muzzle, [px, py, pz]) / speed + lat
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
        spin = float(FLAGS["spin"])
        if self.jet_scan or (spin > 0 and self.hunt and (t % spin) < 2.0):
            # nothing known nearby: turn the torso all the way round every few seconds, so we spot weapons, crates
            # and anyone coming up behind us (vision is a 120 degree wedge; the legs keep walking where they were)
            return me["aim"][0] + 0.35
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
            if p["kind"] in held or (self.hunt and p["kind"] == "mortar"):
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
