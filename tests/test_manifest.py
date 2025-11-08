import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_produce_artifact_manifest_creates_manifest():
    # tests/ is one level below the repository root
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "produce_artifact_manifest.py"
    assert script.exists(), f"manifest script not found at {script}"

    workdir = repo_root / "tmp_manifest_test"
    # ensure clean slate
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()

    try:
        d1 = workdir / "d1"
        d2 = workdir / "d2"
        d1.mkdir()
        d2.mkdir()
        f1 = d1 / "a.txt"
        f2 = d2 / "b.bin"
        f1.write_text("hello")
        f2.write_bytes(b"\x00\x01")

        out_manifest = workdir / "manifest.json"

        # Run the manifest tool with cwd=workdir so relative paths are used
        cmd = [sys.executable, str(script), "--out", str(out_manifest), "--paths", str(d1), str(d2)]
        subprocess.check_call(cmd, cwd=workdir)

        assert out_manifest.exists(), "manifest was not created"
        data = json.loads(out_manifest.read_text())
        assert data.get("num_files") == 2
        rels = [f.get("relpath") for f in data.get("files", [])]
        # Ensure our files are listed (relative paths should include the d1/d2 components)
        assert any("d1/a.txt" in r or r.endswith("d1/a.txt") for r in rels)
        assert any("d2/b.bin" in r or r.endswith("d2/b.bin") for r in rels)

    finally:
        # cleanup
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass
