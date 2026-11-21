"""Static arena map: terrain heights, building prisms, and fast line-of-sight tests.

Coordinates follow protocol.md section 5: x east, y north, z up, meters.
Map format follows protocol.md section 11 (v2/world.json + v2/terrain.i16).
"""
from __future__ import annotations

import json
import math
import os
import sys
from array import array
from typing import List, Optional, Tuple


class Prism:
    """A solid that is not terrain: a polygon footprint (with holes) extruded from base to top."""

    __slots__ = ("kind", "base", "top", "ring", "holes", "edges", "minx", "miny", "maxx", "maxy")

    def __init__(self, raw: dict) -> None:
        self.kind = raw.get("kind", "building")
        self.base = float(raw["base"])
        self.top = float(raw["top"])
        self.ring = _pairs(raw["ring"])
        self.holes = [_pairs(h) for h in (raw.get("holes") or []) if len(h) >= 6]
        self.edges: List[Tuple[float, float, float, float]] = []
        for poly in [self.ring, *self.holes]:
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                self.edges.append((x1, y1, x2, y2))
        xs = [p[0] for p in self.ring]
        ys = [p[1] for p in self.ring]
        self.minx, self.maxx, self.miny, self.maxy = min(xs), max(xs), min(ys), max(ys)

    def contains_xy(self, x: float, y: float) -> bool:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        if not _in_poly(self.ring, x, y):
            return False
        for h in self.holes:
            if _in_poly(h, x, y):
                return False
        return True

    def blocks_segment(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float,
                       ignore_z: bool = False) -> bool:
        """True if the 3D segment a->b passes through this prism's solid (with ignore_z: its footprint)."""
        if not ignore_z and (min(az, bz) > self.top or max(az, bz) < self.base):
            return False
        if max(ax, bx) < self.minx or min(ax, bx) > self.maxx or max(ay, by) < self.miny or min(ay, by) > self.maxy:
            return False
        dx, dy = bx - ax, by - ay
        ts = [0.0, 1.0]
        for x1, y1, x2, y2 in self.edges:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if den == 0.0:
                continue
            qx, qy = x1 - ax, y1 - ay
            t = (qx * ey - qy * ex) / den
            if t <= 0.0 or t >= 1.0:
                continue
            u = (qx * dy - qy * dx) / den
            if 0.0 <= u <= 1.0:
                ts.append(t)
        ts.sort()
        dz = bz - az
        for i in range(len(ts) - 1):
            t0, t1 = ts[i], ts[i + 1]
            if t1 - t0 < 1e-9:
                continue
            tm = 0.5 * (t0 + t1)
            if self.contains_xy(ax + dx * tm, ay + dy * tm):
                if ignore_z:
                    return True
                z0, z1 = az + dz * t0, az + dz * t1
                if max(z0, z1) >= self.base and min(z0, z1) <= self.top:
                    return True
        return False


def _pairs(flat: List[float]) -> List[Tuple[float, float]]:
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]


def _in_poly(poly: List[Tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class World:
    CELL = 16.0  # spatial-index cell size in meters

    def __init__(self, map_dir: str, files: Optional[List[str]] = None) -> None:
        world_rel = "v2/world.json"
        terrain_rel = None
        for f in files or []:
            if f.endswith(".json"):
                world_rel = f
            elif f.endswith(".i16"):
                terrain_rel = f
        world_path = os.path.join(map_dir, world_rel)
        with open(world_path, "r", encoding="utf-8") as fh:
            w = json.load(fh)

        b = w["bounds"]
        self.minx, self.maxx, self.miny, self.maxy = b["minX"], b["maxX"], b["minY"], b["maxY"]

        t = w["terrain"]
        self.tx0, self.ty0 = float(t["minX"]), float(t["minY"])
        self.tstep = float(t["step"])
        self.tnx, self.tny = int(t["nx"]), int(t["ny"])
        self.tscale = float(t.get("scale", 0.01))
        if terrain_rel is None:
            terrain_rel = os.path.join(os.path.dirname(world_rel), t.get("file", "terrain.i16"))
        heights = array("h")
        with open(os.path.join(map_dir, terrain_rel), "rb") as fh:
            heights.frombytes(fh.read())
        if sys.byteorder != "little":
            heights.byteswap()
        self.heights = heights

        self.prisms = [Prism(p) for p in w["prisms"] if len(p.get("ring", [])) >= 6]

        # Spatial index: every prism is listed in each cell its bounding box touches.
        self.gx0, self.gy0 = self.minx - self.CELL, self.miny - self.CELL
        self.gnx = int((self.maxx - self.gx0) / self.CELL) + 3
        self.gny = int((self.maxy - self.gy0) / self.CELL) + 3
        self.grid: List[List[int]] = [[] for _ in range(self.gnx * self.gny)]
        for idx, p in enumerate(self.prisms):
            i0, j0 = self._cell(p.minx, p.miny)
            i1, j1 = self._cell(p.maxx, p.maxy)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    self.grid[j * self.gnx + i].append(idx)

    # ---------- terrain ----------

    def ground(self, x: float, y: float) -> float:
        """Terrain height (bilinear), ignoring buildings."""
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = int(math.floor(fx))
        j = int(math.floor(fy))
        if i < 0:
            i, fx = 0, 0.0
        elif i > self.tnx - 2:
            i, fx = self.tnx - 2, float(self.tnx - 1)
        if j < 0:
            j, fy = 0, 0.0
        elif j > self.tny - 2:
            j, fy = self.tny - 2, float(self.tny - 1)
        ax = fx - i
        ay = fy - j
        k = j * self.tnx + i
        h = self.heights
        h0 = h[k] * (1 - ax) + h[k + 1] * ax
        h1 = h[k + self.tnx] * (1 - ax) + h[k + self.tnx + 1] * ax
        return (h0 * (1 - ay) + h1 * ay) * self.tscale

    # ---------- spatial index ----------

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.gx0) / self.CELL)
        j = int((y - self.gy0) / self.CELL)
        return min(max(i, 0), self.gnx - 1), min(max(j, 0), self.gny - 1)

    def _cells_on_segment(self, ax: float, ay: float, bx: float, by: float) -> set:
        """Prism indices in every grid cell the 2D segment crosses (Amanatides-Woo walk)."""
        out: set = set()
        i, j = self._cell(ax, ay)
        ie, je = self._cell(bx, by)
        dx, dy = bx - ax, by - ay
        step_i = 1 if dx > 0 else -1
        step_j = 1 if dy > 0 else -1
        if dx != 0:
            nxt = self.gx0 + (i + (1 if dx > 0 else 0)) * self.CELL
            t_max_x = (nxt - ax) / dx
            t_dx = self.CELL / abs(dx)
        else:
            t_max_x = t_dx = math.inf
        if dy != 0:
            nyt = self.gy0 + (j + (1 if dy > 0 else 0)) * self.CELL
            t_max_y = (nyt - ay) / dy
            t_dy = self.CELL / abs(dy)
        else:
            t_max_y = t_dy = math.inf
        grid, gnx = self.grid, self.gnx
        for _ in range(self.gnx + self.gny + 4):
            out.update(grid[j * gnx + i])
            if i == ie and j == je:
                break
            if t_max_x < t_max_y:
                t_max_x += t_dx
                i += step_i
                if i < 0 or i >= self.gnx:
                    break
            else:
                t_max_y += t_dy
                j += step_j
                if j < 0 or j >= self.gny:
                    break
        return out

    # ---------- queries ----------

    def building_between(self, a: List[float], b: List[float]) -> bool:
        """True if any building, stadium or bridge solid blocks the straight line a->b."""
        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[0], b[1], b[2]
        for idx in self._cells_on_segment(ax, ay, bx, by):
            if self.prisms[idx].blocks_segment(ax, ay, az, bx, by, bz):
                return True
        return False

    def terrain_between(self, a: List[float], b: List[float], step: float = 3.0) -> bool:
        """True if the ground rises above the line a->b somewhere between the ends."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = int(math.hypot(dx, dy) / step)
        for k in range(1, n):
            t = k / n
            if a[2] + dz * t < self.ground(a[0] + dx * t, a[1] + dy * t) - 0.3:
                return True
        return False

    def clear_line(self, a: List[float], b: List[float]) -> bool:
        """Line of fire / sight ignoring trees (trees never stop shots)."""
        return not self.building_between(a, b) and not self.terrain_between(a, b)

    def wall_ahead(self, x: float, y: float, z: float, yaw: float, dist: float,
                   step: float = 1.2, head: float = 9.0, half_width: float = 2.8) -> bool:
        """True if walking `dist` meters along `yaw` from feet (x, y, z) runs into a solid.

        Probes the center line and both shoulders of the mech (radius 2.5 m, plus a margin) just above
        step-up height (anything taller than 1.2 m is a wall) and just below head height (9 m)."""
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = x + c * dist, y + s * dist
        if ex < self.minx + 4 or ex > self.maxx - 4 or ey < self.miny + 4 or ey > self.maxy - 4:
            return True
        px, py = -s * half_width, c * half_width
        for zz in (z + step + 0.1, z + head - 0.2):
            for ox, oy in ((0.0, 0.0), (px, py), (-px, -py)):
                if self.building_between([x + ox, y + oy, zz], [ex + ox, ey + oy, zz]):
                    return True
        # steep ground counts as a wall too
        rise = self.ground(ex, ey) - max(z, self.ground(x, y))
        return rise > dist * 1.1 and rise > 1.5

    def obstacle_top(self, x: float, y: float, z: float, yaw: float, dist: float, half_width: float = 2.8) -> Optional[float]:
        """Height of the highest solid in the way of walking `dist` along `yaw` from feet (x, y, z), or None."""
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = x + c * dist, y + s * dist
        px, py = -s * half_width, c * half_width
        top = None
        for zz in (z + 1.3, z + 8.8):
            for ox, oy in ((0.0, 0.0), (px, py), (-px, -py)):
                a = [x + ox, y + oy, zz]
                b = [ex + ox, ey + oy, zz]
                for idx in self._cells_on_segment(a[0], a[1], b[0], b[1]):
                    p = self.prisms[idx]
                    if (top is None or p.top > top) and p.blocks_segment(a[0], a[1], a[2], b[0], b[1], b[2]):
                        top = p.top
        return top

    def max_top_along(self, ax: float, ay: float, bx: float, by: float) -> float:
        """Highest solid top whose footprint the 2D segment crosses (-inf if none): can we jet over this line?"""
        top = -math.inf
        for idx in self._cells_on_segment(ax, ay, bx, by):
            p = self.prisms[idx]
            if p.top > top and p.blocks_segment(ax, ay, 0.0, bx, by, 0.0, ignore_z=True):
                top = p.top
        return top

    def inside_solid(self, x: float, y: float, z: float) -> bool:
        i, j = self._cell(x, y)
        for idx in self.grid[j * self.gnx + i]:
            p = self.prisms[idx]
            if p.base <= z <= p.top and p.contains_xy(x, y):
                return True
        return False
