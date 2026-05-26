"""Informal test: process the KIRI fixture OBJ bundle through process_obj_bundle.

Run with: .venv/bin/python test_obj.py
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import trimesh

from processor import process_obj_bundle

FIXTURE_DIR = Path(__file__).parent / ".context" / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> int:
    obj_path = FIXTURE_DIR / "3DModel.obj"
    mtl_path = FIXTURE_DIR / "3DModel.mtl"
    jpg_path = FIXTURE_DIR / "3DModel.jpg"

    if not all(p.exists() for p in (obj_path, mtl_path, jpg_path)):
        print(f"Fixture missing under {FIXTURE_DIR}", file=sys.stderr)
        return 1

    obj_bytes = obj_path.read_bytes()
    mtl_bytes = mtl_path.read_bytes()
    textures = {"3DModel.jpg": jpg_path.read_bytes()}

    print(f"Input OBJ: {len(obj_bytes) / 1024 / 1024:.2f} MB")
    print(f"Input MTL: {len(mtl_bytes)} bytes")
    print(f"Input texture: {len(textures['3DModel.jpg']) / 1024 / 1024:.2f} MB")

    t0 = time.perf_counter()
    glb_bytes, stats = process_obj_bundle(obj_bytes, mtl_bytes, textures, "modelo_kiri")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_path = OUTPUT_DIR / "kiri_textured.glb"
    out_path.write_bytes(glb_bytes)

    print()
    print(f"GLB: {stats.glb_size_bytes / 1024 / 1024:.2f} MB  ({elapsed_ms:.0f} ms)")
    print(f"Triangles: {stats.total_input_triangles:,} -> {stats.total_output_triangles:,}")

    # Reload and assert texture survived + dimensions are now in mm.
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    assert isinstance(scene, trimesh.Scene), f"expected Scene, got {type(scene)}"

    geoms = list(scene.geometry.items())
    assert len(geoms) == 1, f"expected 1 mesh, got {len(geoms)}"
    name, mesh = geoms[0]

    bounds = mesh.bounds
    size_mm = bounds[1] - bounds[0]
    print(f"Reloaded '{name}': bounds size = {size_mm} mm")

    # The fixture is ~85 cm tall; after the meters→mm scale that's ~850 mm.
    # If we ever break the scale, this assert catches it.
    assert size_mm[1] > 500, f"expected ~850 mm tall, got {size_mm[1]} — scale broken"
    assert size_mm[1] < 1200, f"expected ~850 mm tall, got {size_mm[1]} — over-scaled"

    mat = mesh.visual.material
    assert hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None, (
        f"texture missing from GLB; material is {type(mat).__name__}"
    )
    print(f"Texture: {mat.baseColorTexture.size}, mode={mat.baseColorTexture.mode}")

    # And check the zip-extraction path: pack the same files into a zip, run
    # the main.py extractor on it.
    print()
    print("--- zip path ---")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("3DModel.obj", obj_bytes)
        zf.writestr("3DModel.mtl", mtl_bytes)
        zf.writestr("3DModel.jpg", textures["3DModel.jpg"])
    zip_bytes = buf.getvalue()
    print(f"Zip bundle: {len(zip_bytes) / 1024 / 1024:.2f} MB")

    import os
    os.environ["DRY_RUN"] = "true"
    from main import _extract_obj_bundle  # noqa: E402

    obj_b, mtl_b, tex, obj_name = _extract_obj_bundle([("bundle.zip", zip_bytes)])
    assert obj_b == obj_bytes, "obj bytes don't match through zip"
    assert mtl_b == mtl_bytes, "mtl bytes don't match through zip"
    assert "3DModel.jpg" in tex, f"texture missing from zip extract: {list(tex)}"
    print(f"Extracted via zip: obj={obj_name}, textures={list(tex)}")

    print()
    print(f"Output: {out_path}")
    print("Arraste em https://gltf-viewer.donmccurdy.com/ — deve mostrar o modelo com textura.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
