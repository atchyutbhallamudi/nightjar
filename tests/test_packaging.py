"""The Docker image must contain every local module the bot imports (the Dockerfile copies files by name)."""
import ast
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_imports(path, seen=None):
    seen = seen if seen is not None else set()
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        for n in names:
            f = os.path.join(ROOT, n + ".py")
            if os.path.exists(f) and n not in seen:
                seen.add(n)
                local_imports(f, seen)
    return seen


class Packaging(unittest.TestCase):
    def test_dockerfile_copies_every_module_bot_imports(self):
        with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
            text = fh.read()
        copied = set()
        for line in text.splitlines():
            m = re.match(r"\s*COPY\s+(.+)\s+\S+\s*$", line)
            if m:
                copied.update(os.path.splitext(os.path.basename(p))[0] for p in m.group(1).split())
        needed = local_imports(os.path.join(ROOT, "bot.py")) | {"bot"}
        missing = sorted(needed - copied)
        self.assertFalse(missing, f"Dockerfile does not COPY: {missing}")

    def test_entrypoint_is_bot(self):
        with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
            self.assertIn('"/bot/bot.py"', fh.read())


if __name__ == "__main__":
    unittest.main()
