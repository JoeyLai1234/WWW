"""
produce_artifact_manifest.py

Scan common artifact folders produced by the demo and write a JSON manifest
with relative paths, sizes (bytes), and sha256 checksums.

Usage:
  python tools/produce_artifact_manifest.py --out manifest.json

This is intentionally small and dependency-free (uses stdlib only).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_PATHS = ["utils", "utils/heat", "images", "heatmap", "heatmap_info"]


def sha256_of_file(p: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def gather(paths, follow_symlinks: bool = False):
    files = []
    cwd = Path.cwd().resolve()
    for base in paths:
        basep = Path(base)
        if not basep.exists():
            continue
        for p in basep.rglob("*"):
            if p.is_file():
                try:
                    size = p.stat().st_size
                except OSError:
                    size = None
                # Make absolute and compute a relative path to cwd when possible
                try:
                    abs_p = p.resolve()
                except Exception:
                    abs_p = p.absolute()
                try:
                    rel = str(abs_p.relative_to(cwd).as_posix())
                except Exception:
                    # fallback to os.path.relpath for robustness
                    rel = os.path.relpath(str(abs_p), str(cwd))
                files.append({
                    "path": str(abs_p.as_posix()),
                    "relpath": rel,
                    "size": size,
                })
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifact_manifest.json", help="output JSON file")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS, help="paths to scan")
    parser.add_argument("--checksums", action="store_true", help="compute sha256 checksums (slower)")
    args = parser.parse_args()

    files = gather(args.paths)
    if args.checksums:
        for f in files:
            p = Path(f["path"])
            try:
                f["sha256"] = sha256_of_file(p)
            except Exception as e:
                f["sha256"] = None
                f["sha256_error"] = str(e)

    manifest = {
        "root": str(Path.cwd()),
        "scanned_paths": args.paths,
        "num_files": len(files),
        "files": files,
    }

    outp = Path(args.out)
    outp.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {outp} ({len(files)} files)")


if __name__ == "__main__":
    main()
