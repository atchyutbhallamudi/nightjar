"""Aiming math: shots fired at the computed aim should actually meet the target."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import Bot  # noqa: E402


def bare_bot() -> Bot:
    b = Bot.__new__(Bot)
    b.cfg = {"mech": {"min_pitch": -0.7, "max_pitch": 1.3}}
    b.mortar_g = 22.0
    return b


def closest_shell_miss(yaw, pitch, start, target, v=72.0, g=22.0):
    x, y, z = start
    vx, vy, vz = v * math.cos(pitch) * math.cos(yaw), v * math.cos(pitch) * math.sin(yaw), v * math.sin(pitch)
    best = 1e9
    dt = 0.0005
    for _ in range(30000):
        x += vx * dt
        y += vy * dt
        vz -= g * dt
        z += vz * dt
        best = min(best, math.dist([x, y, z], target))
    return best


class Aim(unittest.TestCase):
    def test_mortar_low_and_high_arcs_land_on_target(self):
        b = bare_bot()
        me = {"pos": [0.0, 0.0, 0.0], "stats": {"height": 9}}
        mortar = {"stats": {"speed": 72, "gravity": 22}}
        for tgt in ([100, 0, 0], [150, 50, 5], [200, 0, 0]):
            for los in (True, False):
                sol = b._mortar_aim(me, mortar, tgt, [0, 0, 0], los)
                if sol is None:
                    continue
                miss = closest_shell_miss(sol[0], sol[1], [0, 0, 9.5], [tgt[0], tgt[1], tgt[2] + 1])
                self.assertLess(miss, 0.5, f"{tgt} los={los} missed by {miss:.2f} m")

    def test_mortar_refuses_impossible_shots(self):
        b = bare_bot()
        me = {"pos": [0.0, 0.0, 0.0], "stats": {"height": 9}}
        self.assertIsNone(b._mortar_aim(me, {"stats": {"speed": 72, "gravity": 22}}, [400, 0, 0], [0, 0, 0], True))

    def test_gun_leads_a_crossing_target(self):
        b = bare_bot()
        yaw, pitch = b._direct_aim([0, 0, 7], [120, 0, 0], [0, 12, 0], 4.5, 252)
        best = 1e9
        for i in range(1, 4000):
            s = i * 0.0005
            bullet = [252 * s * math.cos(pitch) * math.cos(yaw), 252 * s * math.cos(pitch) * math.sin(yaw),
                      7 + 252 * s * math.sin(pitch)]
            best = min(best, math.dist(bullet, [120, 12 * (s + 0.05), 4.5]))  # shot leaves ~0.05 s after the observation
        self.assertLess(best, 0.3)


if __name__ == "__main__":
    unittest.main()
