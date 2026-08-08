"""swbd_* must import as package modules without touching the filesystem."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_swbd_modules_import_as_package_modules():
    for mod in ("src.swbd_parse", "src.swbd_convert"):
        importlib.import_module(mod)


def test_importing_swbd_convert_creates_no_directories(tmp_path):
    """Import must be side-effect free: run it from an empty cwd and check."""
    code = "import src.swbd_convert"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**dict(PYTHONPATH=str(ROOT)), "PATH": "/usr/bin:/bin"},
        check=True,
    )
    created = [p.name for p in tmp_path.iterdir()]
    assert created == [], f"import created {created} in the working directory"


def test_no_module_level_mkdir_calls():
    tree = ast.parse((ROOT / "src/swbd_convert.py").read_text())
    for node in tree.body:  # module level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a mkdir in a function body runs on call, not on import
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "attr", "") == "mkdir":
                raise AssertionError(
                    f"module-level mkdir at line {sub.lineno}; move it into main()"
                )
