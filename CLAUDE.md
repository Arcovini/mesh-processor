# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mesh-processor` is the backend service for **medCaseViewer** (https://biodesignlab.com.br) — a 3D surgical planning tool for the Brazilian healthcare market.

Its single responsibility: receive raw STL files (typically generated from medical imaging segmentation), optimize them for web viewing, and publish them to Sketchfab so they can be loaded by the existing static viewer at `https://biodesignlab.com.br/case/?id=<UID>`.

This service is **Sprint 1** of a broader migration plan. See "Roadmap" below — design decisions here exist to make Sprints 2 and 3 painless. Do not collapse abstractions that exist for that reason.

A single `/upload` request accepts **multiple STLs** at once (one clinical case = N anatomical structures) and produces **one** GLB containing each STL as a named mesh node. The viewer's `getNodeMap` uses those names to render per-structure toggles and opacity sliders.

## Development

### Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in SKETCHFAB_TOKEN (or set DRY_RUN=true)
set -a && source .env && set +a
uvicorn main:app --reload --port 8000
```

For most dev work, `DRY_RUN=true` in `.env` is enough — see "DRY_RUN" below.

### Testing the upload endpoint

Multi-file form (normal case — one clinical case has several structures):

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@artery.stl" \
  -F "files=@vein.stl" \
  -F "files=@kidney.stl"
```

The `name` form field **does not exist** — the model name on Sketchfab is auto-generated as `YYYY-MM-DD-<12 random digits>`, a placeholder identifier until a DB layer is introduced. The original STL filenames are cleaned (see `clean_mesh_names` in `main.py`) and become the GLB node names that the viewer displays as toggle labels.

### Building/running with Docker

```bash
docker build -t mesh-processor .
docker run -p 8000:8000 -e DRY_RUN=true mesh-processor
```

The Dockerfile installs `build-essential` because `fast-simplification` compiles a C++ extension on `pip install` and there's no prebuilt wheel for `linux/aarch64`.

### Deployment

Hosted on Railway. Push to `main` triggers auto-deploy. Railway auto-detects the Dockerfile and uses it. Required env vars (set in the Railway dashboard):
- `SKETCHFAB_TOKEN` — Sketchfab API token (secret)
- `VIEWER_BASE` — defaults to `https://biodesignlab.com.br/case/`
- `PORT` — set automatically by Railway
- `DRY_RUN` — leave unset (or `false`) in production

## Architecture

### Project structure

```
mesh-processor/
├── main.py          # FastAPI app — /upload, /status, /health. Pure orchestration.
├── processor.py     # STL(s) → scene → multi-mesh GLB (pure, testable, no I/O)
├── sketchfab.py     # Sketchfab API client. Thin. DRY_RUN short-circuit inside.
├── test_processor.py  # Informal smoke test against real STLs (not pytest)
├── .env.example     # Template for SKETCHFAB_TOKEN, VIEWER_BASE, DRY_RUN
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

### Critical separation: processor vs destination

`processor.py` knows nothing about Sketchfab. `sketchfab.py` knows nothing about meshes. This is intentional. In Sprint 2 a sibling `r2.py` will be added to upload to Cloudflare R2 in parallel; in Sprint 3 Sketchfab will be removed entirely. **Do not couple them.**

The same separation has a second payoff: `processor.py` returns `bytes`, not a file path. In Sprint 2 those bytes go to **two** destinations simultaneously (Sketchfab + R2) with zero processor changes. Keeping intermediates in memory costs ~10MB per request and saves the duplication.

### Pure vs effectful split

`processor.py` is a **pure** module — `bytes in → bytes out`, no I/O, no env vars, no network. Call it 1000 times with the same input and it returns the same output. Testable without any setup.

`sketchfab.py` is **effectful** — it mutates the world (creates models on Sketchfab, consumes quota). That's why it has a DRY_RUN short-circuit: we want to exercise every layer above it (endpoint, CORS, the eventual upload page) without actually triggering effects until we're ready.

The rule: separate computation from side effects at file boundaries. Testing the pure part is free; testing the effectful part costs slots/dollars/time and should be done sparingly.

### Mesh processing decisions (medical context)

These defaults exist because this is **medical/surgical data**, not generic 3D content:

- **Target triangle count: 300,000 per mesh** (not per scene). Preserves anatomical detail (fractures, calcifications, vessel branches) while keeping each structure lean. Configurable per request via `target_triangles` form field.
- **Decimation algorithm: `fast_simplification` (quadric edge collapse).** Chosen over `pymeshlab` (heavy install, GPL) and `trimesh.simplify_quadric_decimation` (slower, less stable on large meshes).
- **No aggressive smoothing.** `trimesh.load(process=True)` does safe cleanup (duplicate vertices, normals). Anything more (Laplacian smoothing, Taubin) can round off clinically relevant features and is **off by default**. If a future request needs it, gate it behind an explicit flag, not a default.
- **Coordinate system: STL is RAS (Z-up), glTF is Y-up.** We apply a fixed `-π/2` rotation around X so each mesh lands upright in the viewer. The rotation is identical for every mesh in a batch, preserving inter-structure spatial relationships (a kidney, its artery, its vein, and a lesion from the same exam stay co-registered).
- **Output format: GLB binary.** Smaller than glTF+bin, single file, native browser support. Sketchfab's preferred format.
- **Per-mesh PBR materials, not vertex colors.** Each structure gets a named `PBRMaterial` (`baseColorFactor` + `roughnessFactor=0.5` + `metallicFactor=0`). Reason: Sketchfab's viewer API (`api.setMaterial`) operates on the *material list*. Without distinct materials, the viewer's opacity slider per structure cannot function. Vertex colors would render visually but would be a single material in the viewer.

### Color assignment (keyword-based with colorblind-safe fallback)

`main.py > clean_mesh_names` strips common prefix/suffix across the batch plus a `_(timestamp)` regex, then transliterates accents. The resulting lowercased name is matched against keywords in `processor.COLORS_BY_KEYWORD`:

| Keyword substring | Color | Meaning |
|---|---|---|
| `art` | `#BD0006` | artéria (dark red) |
| `veia` | `#458DE7` | veia (blue) |
| `rim` | `#BA5531` | rim (brown-orange) |
| `lesao` | `#08E700` | lesão (bright green) |
| `pele` | `#C4908E` | pele (skin pink) |

Non-matched names cycle through an IBM Colorblind Safe palette (`FALLBACK_COLORS`) by index — deterministic, so the same name consistently gets the same color. To add/change a clinical category, edit `COLORS_BY_KEYWORD`; don't touch the fallback palette lightly — reordering it reshuffles colors for unmatched cases.

### Required: force vertex-normal compute after transforms

`apply_transform` invalidates trimesh's cached normals. The GLB exporter only writes the `NORMAL` attribute if normals exist on the mesh at export time. Without `NORMAL`, viewers render flat-shaded (visible triangle facets).

**The fix in `process_stls`:** after `mesh.apply_transform(...)`, access `_ = mesh.vertex_normals` to force lazy recompute. This single line is what makes the viewer render smooth. `trimesh.Trimesh(..., process=True)` handles the post-decimation case; the access handles the pass-through case. Both are needed.

`scipy` is a transitive dep of trimesh for this compute — it's pinned in `requirements.txt`. Without scipy, the `vertex_normals` access raises a swallowed `ModuleNotFoundError` and the GLB comes out without normals.

### Input limits

- **Max STL size: 60MB.** Set in `main.py`. STLs larger than this should not exist in our pipeline (segmentation output is capped upstream). If they do, fail loudly — silently truncating clinical data is dangerous.
- **STL only for now.** OBJ/PLY/etc. could be added but are not in scope.

### Sketchfab integration notes

- Upload returns a `uid` immediately, but processing on Sketchfab's side takes 30-90s for large models (<5s for small multi-mesh cases). The frontend polls `/status/{uid}` until it returns `ready: true`.
- Always include `'source': 'biodesignlab'` per Sketchfab's developer guidelines (they use it for internal tracking).
- API token is account-wide — never expose it to the frontend. All Sketchfab calls go through this service.
- Sketchfab plan file-size limits: Basic 100MB/file, Pro 200MB, Premium 500MB. Our 60MB cap stays well below.

**Monthly upload cap bypass via `isDownloadable=true`.** Sketchfab's pricing page states: *"A model that is downloadable doesn't count against this limit."* Setting `isDownloadable: "true"` on the upload means the model does not consume one of the Basic plan's 10 monthly slots. biodesignlab is on the Basic (free) tier; without this flag, ~10 clinical cases per month would exhaust the account. **Trade-off:** anyone with the model's URL can download the GLB. For biodesignlab this is acceptable because segmented anatomical STLs contain no PHI. If ever storing identifiable data, revisit.

**Working upload field names** (confirmed by real upload on 2026-04-23):

| Form field | Value | Notes |
|---|---|---|
| `modelFile` | `(filename, bytes, "model/gltf-binary")` | Multipart file. Field name is *exactly* `modelFile` — not `file`. |
| `name` | auto-generated `YYYY-MM-DD-<12 random digits>` | Model name on sketchfab.com |
| `source` | `"biodesignlab"` | per Sketchfab dev guidelines |
| `isPublished` | `"true"` | string, not bool |
| `private` | `"false"` | Basic plan can't do private anyway; we keep it explicit |
| `isDownloadable` | `"true"` | **critical** — bypasses the 10/mo cap |

Response on 201 Created: `{"uid": "..."}`. Status endpoint response: `status.processing` is `SUCCEEDED` when ready.

### CORS

Only `https://biodesignlab.com.br` and `http://localhost:5501` (Live Server in the viewer repo) are allowed. Add new origins explicitly — do not use `*`.

## Code patterns

- **FastAPI with type hints.** Use `Form()`, `UploadFile`, and Pydantic models for request validation. Return plain dicts for responses (FastAPI handles serialization).
- **Errors as HTTPException with clear messages.** The frontend surfaces these directly to clinicians, so keep them human-readable in Portuguese where user-facing. Ex: `raise HTTPException(400, "STL inválido ou corrompido: ...")`.
- **No background workers / queues yet.** Processing happens synchronously in the request. Multi-STL cases process in <1s plus the Sketchfab upload RTT (~1-2s). If we add larger inputs or batch processing, revisit with Celery + Redis.
- **Stateless service.** No database. Sketchfab is the source of truth for uploaded models. Sprint 2 will add R2 as a parallel store but still no DB until metadata requirements appear.
- **Pure functions in `processor.py`.** Takes bytes, returns bytes + stats. No I/O, no env vars. Makes it trivially testable and reusable.
- **Thin clients** (currently just `sketchfab.py`; future `r2.py` will follow the same shape). Only HTTP + response parsing. No retry loops, no caching, no polling. Orchestration (retry, concurrency, cross-destination logic) lives in `main.py`. A thin client is cheap to delete when Sprint 3 removes Sketchfab.

### DRY_RUN pattern

The `DRY_RUN=true` env var short-circuits `sketchfab.upload_model` / `sketchfab.get_status` inside the same file, a few lines from the real call. It returns a known-working Sketchfab UID by default (configurable via `DRY_RUN_UID` env), so the resulting `viewer_url` actually resolves in the browser — perfect for testing the upload page, CORS, and the full UX loop without burning slots.

`DRY_RUN=true` also lets the service boot without `SKETCHFAB_TOKEN` — useful for CI and for hand-off to new contributors who don't have a token yet. Production must never set `DRY_RUN=true`.

The fake/real paths living in the same file is deliberate: if the real Sketchfab API shape changes, the DRY_RUN branch is right there, three lines away, and gets updated in the same diff. A separate mock file would drift silently.

### Error handling inside `processor.py`

`_load_and_decimate` catches **any** exception from `trimesh.load` and re-raises as `ValueError("STL inválido ou corrompido: ...")`. Reason: trimesh raises a variety of exception types depending on what's wrong with the STL (ModuleNotFoundError for missing optional deps, KeyError for malformed internal refs, etc.). The contract guaranteed by `processor.py` to `main.py` is: "bad input → ValueError, always." This lets `main.py` catch exactly that and return HTTP 400 without guessing.

Transitive deps worth knowing about:
- `scipy` — needed by trimesh to compute per-vertex normals (see smooth-shading note above).
- `chardet` — needed by trimesh's STL ASCII fallback when the binary parse fails. Without it, feeding random bytes to `trimesh.load` raises `ModuleNotFoundError` (silently without our `except`).

## Roadmap (do not break these paths)

- **Sprint 1 (current):** STL → optimized GLB → Sketchfab. Viewer URL pattern unchanged.
- **Sprint 2:** Add `r2.py`. After Sketchfab upload succeeds, also push GLB to Cloudflare R2 at `cases/{uid}.glb`. Same UID, two storage backends.
- **Sprint 3:** Rewrite `medCaseViewer/case/` from Sketchfab iframe to native Three.js + GLTFLoader reading from R2. Sketchfab becomes optional/removed. Public URL (`?id=...`) stays identical — clinicians with old links keep working.
- **Future (after AI pipeline):** A separate `ai-segmentation` service will produce STLs from DICOM and POST them to this service's `/upload` endpoint. This service does not need to know whether the STL came from a human upload or an AI run.

The canonical, cross-service version of this roadmap lives in the workspace-level `CLAUDE.md` (one folder up). If the two ever disagree, that one wins.

## What this service is NOT responsible for

- DICOM parsing or AI segmentation (separate service, not built yet).
- Authentication or user accounts (out of scope; unguessable Sketchfab UIDs are the access control for now).
- Storing case metadata (patient name, exam date, etc.) — when this is needed, add a Postgres on Railway and a separate `cases` service. Do not bolt it onto this one.
- Frontend rendering / Three.js / measurement tools (lives in the `medCaseViewer` repo).
