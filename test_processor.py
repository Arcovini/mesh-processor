"""Informal test: process all Reynaldo STLs as a single multi-mesh GLB.

Run with: .venv/bin/python test_processor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from processor import process_stls

SAMPLES_DIR = Path("/Users/viniciusarcoverde/Downloads/Reynaldo Real Martins Junior 2")
OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> int:
    stls = sorted(SAMPLES_DIR.glob("*.stl"))
    if not stls:
        print(f"No STLs found in {SAMPLES_DIR}", file=sys.stderr)
        return 1

    files = [(p.stem, p.read_bytes()) for p in stls]
    print(f"Building combined GLB from {len(files)} STLs:")
    for name, data in files:
        print(f"  - {name} ({len(data)/1024:.1f} KB)")

    t0 = time.perf_counter()
    glb_bytes, stats = process_stls(files)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_path = OUTPUT_DIR / "reynaldo_combined.glb"
    out_path.write_bytes(glb_bytes)

    print(f"\nCombined GLB: {stats.glb_size_bytes/1024:.1f} KB  ({elapsed_ms:.0f} ms)")
    print(f"Total triangles: {stats.total_input_triangles} -> {stats.total_output_triangles}")
    print("\nPer-mesh breakdown:")
    for m in stats.meshes:
        mark = "(decimated)" if m.decimated else "(pass-through)"
        print(
            f"  {m.name}\n"
            f"    {m.input_triangles} -> {m.output_triangles} tris  {mark}"
        )
    print(f"\nOutput: {out_path}")
    print("Arraste o arquivo acima em https://gltf-viewer.donmccurdy.com/")
    print("No painel 'Scene' da direita, devem aparecer 4 nós nomeados — um por STL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
