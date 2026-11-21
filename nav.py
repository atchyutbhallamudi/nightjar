"""Route planning on the static map: a walkable grid plus A* that can be spread over several decisions.

The grid marks 4 m cells a mech (radius 2.5 m) cannot stand in at ground level: walls of buildings,
stadium and low bridges, and the map edge. Bridges high enough to walk under stay open, and steep ground
is handled per step. Built once at `hello` (about a second) from world.py's prisms.
"""
from __future__ import annotations

import heapq
import math
import time
from array import array
from typing import List, Optional, Tuple

SQRT2 = math.sqrt(2.0)


class NavGrid:
    def __init__(self, world, cell: float = 4.0, clearance: float = 3.5, head: float = 9.0, step: float = 1.2,
                 max_slope_deg: float = 50.0) -> None:
        self.w = world
        self.cell = cell
        self.x0 = world.minx
        self.y0 = world.miny
        self.nx = int((world.maxx - world.minx) / cell) + 1
        self.ny = int((world.maxy - world.miny) / cell) + 1
        n = self.nx * self.ny
        self.ground = [0.0] * n
        for j in range(self.ny):
            cy = self.y0 + (j + 0.5) * cell
            row = j * self.nx
            for i in range(self.nx):
                self.ground[row + i] = world.ground(self.x0 + (i + 0.5) * cell, cy)
        blocked = bytearray(n)
        edge = int(math.ceil(clearance / cell))
        for j in range(self.ny):
            for i in range(self.nx):
                if i < edge or j < edge or i >= self.nx - edge or j >= self.ny - edge:
                    blocked[j * self.nx + i] = 1
        offs = [(0.0, 0.0)] + [(clearance * math.cos(a), clearance * math.sin(a))
                               for a in [k * math.pi / 4 for k in range(8)]]
        for p in world.prisms:
            i0 = max(0, int((p.minx - clearance - self.x0) / cell))
            i1 = min(self.nx - 1, int((p.maxx + clearance - self.x0) / cell))
            j0 = max(0, int((p.miny - clearance - self.y0) / cell))
            j1 = min(self.ny - 1, int((p.maxy + clearance - self.y0) / cell))
            for j in range(j0, j1 + 1):
                cy = self.y0 + (j + 0.5) * cell
                for i in range(i0, i1 + 1):
                    k = j * self.nx + i
                    if blocked[k]:
                        continue
                    g = self.ground[k]
                    # only solids between a step and head height stop a walking mech
                    if p.base >= g + head or p.top <= g + step:
                        continue
                    cx = self.x0 + (i + 0.5) * cell
                    for ox, oy in offs:
                        if p.contains_xy(cx + ox, cy + oy):
                            blocked[k] = 1
                            break
        # ground rising faster than the walkable slope between neighboring cells counts as a wall
        self.max_rise = cell * math.tan(math.radians(max_slope_deg - 2.0))
        self._mark_steep(blocked, max_slope_deg)
        self.blocked = blocked
        self._search = None

    def _mark_steep(self, blocked: bytearray, max_slope_deg: float) -> None:
        """Block cells containing any 1 m terrain step steeper than the walkable slope (terrain walls such as
        the stadium banks), using the raw 1 m height grid."""
        w = self.w
        h, tnx, tny = w.heights, w.tnx, w.tny
        lim = math.tan(math.radians(max_slope_deg)) * w.tstep / w.tscale  # in raw height units per 1 m step
        steep = bytearray(tnx * tny)
        for j in range(tny - 1):
            row = j * tnx
            nrow = row + tnx
            for i in range(tnx - 1):
                k = row + i
                v = h[k]
                if abs(h[k + 1] - v) > lim or abs(h[nrow + i] - v) > lim:
                    steep[k] = 1
        per = max(1, int(round(self.cell / w.tstep)))
        for j in range(self.ny):
            for i in range(self.nx):
                k = j * self.nx + i
                if blocked[k]:
                    continue
                x0 = self.x0 + i * self.cell
                y0 = self.y0 + j * self.cell
                ti0 = int((x0 - w.tx0) / w.tstep)
                tj0 = int((y0 - w.ty0) / w.tstep)
                hit = False
                for tj in range(max(0, tj0), min(tny - 1, tj0 + per)):
                    base = tj * tnx
                    for ti in range(max(0, ti0), min(tnx - 1, ti0 + per)):
                        if steep[base + ti]:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    blocked[k] = 1

    # ---------- cells ----------

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.x0) / self.cell)
        j = int((y - self.y0) / self.cell)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def center(self, i: int, j: int) -> Tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[j * self.nx + i]

    def nearest_free(self, i: int, j: int, max_r: int = 8) -> Optional[Tuple[int, int]]:
        if self.free(i, j):
            return i, j
        for r in range(1, max_r + 1):
            best = None
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    if self.free(i + di, j + dj):
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, i + di, j + dj)
            if best:
                return best[1], best[2]
        return None

    def line_free(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if every grid cell along the straight segment is walkable."""
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / (self.cell * 0.5)))
        prev_g = None
        for k in range(n + 1):
            t = k / n
            i, j = self.to_cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            kk = j * self.nx + i
            if self.blocked[kk]:
                return False
            g = self.ground[kk]
            if prev_g is not None and g - prev_g > self.max_rise:
                return False
            prev_g = g
        return True

    # ---------- A* spread over calls ----------

    def start(self, sx: float, sy: float, gx: float, gy: float,
              keep: Optional[Tuple[float, float, float]] = None, outside_cost: float = 6.0) -> bool:
        """Begin a search. `keep` = (cx, cy, r): cells outside that circle (the zone) cost `outside_cost`
        times more, so routes stay inside it unless there is no other way."""
        s = self.nearest_free(*self.to_cell(sx, sy))
        g = self.nearest_free(*self.to_cell(gx, gy), max_r=12)
        if s is None or g is None:
            self._search = None
            return False
        nx = self.nx
        sk, gk = s[1] * nx + s[0], g[1] * nx + g[0]
        kc = None
        if keep is not None:
            # circle in cell units, so the inner loop stays cheap
            kc = ((keep[0] - self.x0) / self.cell - 0.5, (keep[1] - self.y0) / self.cell - 0.5,
                  (keep[2] / self.cell) ** 2, outside_cost)
        # flat arrays (not dicts) keep the search from making garbage that triggers slow collector pauses
        n = self.nx * self.ny
        cost = array("d", [1e18]) * n
        came = array("l", [-2]) * n
        cost[sk] = 0.0
        came[sk] = -1
        self._search = {
            "goal": gk, "gi": g[0], "gj": g[1], "open": [(0.0, 0.0, sk)], "came": came, "cost": cost,
            "keep": kc,
        }
        return True

    def step(self, budget_s: float) -> Optional[List[Tuple[float, float]]]:
        """Continue the search for up to budget_s. Returns the path (cell centers) once found,
        [] if there is no path, or None if it needs more time."""
        s = self._search
        if s is None:
            return []
        nx, ny, blocked, ground, rise = self.nx, self.ny, self.blocked, self.ground, self.max_rise
        gi, gj, gk = s["gi"], s["gj"], s["goal"]
        openh, came, cost = s["open"], s["came"], s["cost"]
        keep = s.get("keep")
        kx, ky, kr2, kcost = keep if keep is not None else (0.0, 0.0, 0.0, 1.0)
        t_end = time.perf_counter() + budget_s
        count = 0
        while openh:
            count += 1
            if count % 256 == 0 and time.perf_counter() > t_end:
                return None
            _, c, k = heapq.heappop(openh)
            if c > cost[k]:
                continue
            if k == gk:
                path = []
                while k != -1:
                    path.append(self.center(k % nx, k // nx))
                    k = came[k]
                path.reverse()
                self._search = None
                return path
            i, j = k % nx, k // nx
            gk0 = ground[k]
            for di, dj, w in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)):
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                    continue
                nk = nj * nx + ni
                if blocked[nk] or ground[nk] - gk0 > rise * w:
                    continue
                if di and dj and (blocked[j * nx + ni] or blocked[nj * nx + i]):
                    continue  # no corner cutting
                nc = c + w
                if keep is not None and (ni - kx) ** 2 + (nj - ky) ** 2 > kr2:
                    nc += w * (kcost - 1.0)
                if nc < cost[nk]:
                    cost[nk] = nc
                    came[nk] = k
                    dx, dy = abs(ni - gi), abs(nj - gj)
                    h = (dx + dy) + (SQRT2 - 2) * min(dx, dy)
                    heapq.heappush(openh, (nc + h * 1.05, nc, nk))
        self._search = None
        return []

    def searching(self) -> bool:
        return self._search is not None

    def smooth(self, path: List[Tuple[float, float]], max_points: int = 60, max_skip: int = 32) -> List[Tuple[float, float]]:
        """Drop waypoints that a straight walk can skip (string pulling), cheaply: from each kept point,
        double the jump while the straight line stays walkable, then binary-search the farthest one.
        Only the first `max_points` cells are smoothed (the rest is re-planned before we get there)."""
        n = len(path)
        if n <= 2:
            return path
        lim = min(n - 1, max_points)
        out = [path[0]]
        i = 0
        while i < lim:
            ok, step = i + 1, 2
            while True:  # gallop
                j = min(i + step, lim, i + max_skip)
                if self.line_free(path[i][0], path[i][1], path[j][0], path[j][1]):
                    ok = j
                    if j == lim or j == i + max_skip:
                        break
                    step *= 2
                else:
                    break
            lo, hi = ok, min(i + step, lim, i + max_skip)  # farthest good lies in [ok, hi)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self.line_free(path[i][0], path[i][1], path[mid][0], path[mid][1]):
                    lo = mid
                else:
                    hi = mid
            out.append(path[lo])
            i = lo
        out.extend(path[lim + 1:])
        return out
