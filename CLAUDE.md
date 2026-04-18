# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mesh-processor` is the backend service for **medCaseViewer** (https://biodesignlab.com.br) — a 3D surgical planning tool for the Brazilian healthcare market.

Its single responsibility: receive raw STL files (typically generated from medical imaging segmentation), optimize them for web viewing, and publish them to Sketchfab so they can be loaded by the existing static viewer at `https://biodesignlab.com.br/case/?id=<UID>`.

This service is **Sprint 1** of a broader migration plan. See "Roadmap" below — design decisions here exist to make Sprints 2 and 3 painless. Do not collapse abstractions that exist for that reason.

## Development

### Running locally

```bash
pip install -r requirements.txt
export SKETCHFAB_TOKEN=<token from sketchfab.com/settings/password>
uvicorn main:app --reload --port 8000
```

### Testing the upload endpoint

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.stl" \
  -F "name=Test case"
```

### Building/running with Docker

```bash
docker build -t mesh-processor .
docker run -p 8000:8000 -e SKETCHFAB_TOKEN=$SKETCHFAB_TOKEN mesh-processor
```

### Deployment

Hosted on Railway. Push to `main` triggers auto-deploy. Required env vars:
- `SKETCHFAB_TOKEN` — Sketchfab API token (secret)
- `VIEWER_BASE` — defaults to `https://biodesignlab.com.br/case/`
- `PORT` — set automatically by Railway

## Architecture

### Project structure

```
mesh-processor/
├── main.py          # FastAPI app, request handling, CORS
├── processor.py     # STL → decimation → GLB conversion (pure, testable)
├── sketchfab.py     # Sketchfab API client (upload + status polling)
├── Dockerfile
└── requirements.txt
```

### Critical separation: processor vs destination

`processor.py` knows nothing about Sketchfab. `sketchfab.py` knows nothing about meshes. This is intentional. In Sprint 2 a sibling `r2.py` will be added to upload to Cloudflare R2 in parallel; in Sprint 3 Sketchfab will be removed entirely. **Do not couple them.**

### Mesh processing decisions (medical context)

These defaults exist because this is **medical/surgical data**, not generic 3D content:

- **Target triangle count: 300,000.** Preserves anatomical detail (fractures, calcifications, vessel branches) while keeping GLB under ~10MB. Configurable per request via `target_triangles` form field.
- **Decimation algorithm: `fast_simplification` (quadric edge collapse).** Chosen over `pymeshlab` (heavy install, GPL) and `trimesh.simplify_quadric_decimation` (slower, less stable on large meshes).
- **No aggressive smoothing.** `trimesh.load(process=True)` does safe cleanup (duplicate vertices, normals). Anything more (Laplacian smoothing, Taubin) can round off clinically relevant features and is **off by default**. If a future request needs it, gate it behind an explicit flag, not a default.
- **Output format: GLB binary.** Smaller than glTF+bin, single file, native browser support. Sketchfab's preferred format.

### Input limits

- **Max STL size: 60MB.** Set in `main.py`. STLs larger than this should not exist in our pipeline (segmentation output is capped upstream). If they do, fail loudly — silently truncating clinical data is dangerous.
- **STL only for now.** OBJ/PLY/etc. could be added but are not in scope.

### Sketchfab integration notes

- Upload returns a `uid` immediately, but processing on Sketchfab's side takes 30-90s. The frontend polls `/status/{uid}` until it returns `ready: true`.
- Always include `'source': 'biodesignlab'` per Sketchfab's developer guidelines (they use it for internal tracking).
- API token is account-wide — never expose it to the frontend. All Sketchfab calls go through this service.
- Sketchfab plan limits: Free 100MB/file, Pro 200MB, Premium 500MB. Our 60MB cap stays well below.

### CORS

Only `https://biodesignlab.com.br` and `http://localhost:5501` (Live Server in the viewer repo) are allowed. Add new origins explicitly — do not use `*`.

## Code patterns

- **FastAPI with type hints.** Use `Form()`, `UploadFile`, and Pydantic models for request validation. Return plain dicts for responses (FastAPI handles serialization).
- **Errors as HTTPException with clear messages.** The frontend surfaces these directly to clinicians, so keep them human-readable in Portuguese where user-facing.
- **No background workers / queues yet.** Processing happens synchronously in the request. STLs at our size cap (60MB → ~300k tris) process in 5-15s, which is acceptable. If we add larger inputs or batch processing, revisit with Celery + Redis.
- **Stateless service.** No database. Sketchfab is the source of truth for uploaded models. Sprint 2 will add R2 as a parallel store but still no DB until metadata requirements appear.
- **Pure functions in `processor.py`.** Takes bytes, returns bytes + stats. No I/O, no env vars. Makes it trivially testable and reusable.

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
