# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mesh-processor` is the backend service for **medCaseViewer** (https://biodesignlab.com.br) — a 3D surgical planning tool for the Brazilian healthcare market.

Its single responsibility: receive raw STL files (typically generated from medical imaging segmentation), optimize them for web viewing, and store them in Cloudflare R2, where the viewer at `https://biodesignlab.com.br/case/?id=<UID>` loads them — along with the case's image exam, if any (see "Image exam").

This service was built in **Sprint 1** (Sketchfab as the only destination), extended in **Sprint 2** (parallel push to Cloudflare R2) and **Sprint 3d** (image exam). Since **Sprint 3c** (2026-09-27) R2 is the only destination: nothing is written to Sketchfab anymore. See "Roadmap" below.

A single `/upload` request accepts **multiple STLs** at once (one clinical case = N anatomical structures) and produces **one** GLB containing each STL as a named mesh node. The viewer's `getNodeMap` uses those names to render per-structure toggles and opacity sliders.

## Development

### Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the R2 credentials and EXAM_WRITE_SECRET (or set DRY_RUN=true)
set -a && source .env && set +a
uvicorn main:app --reload --port 8000
```

For most dev work, `DRY_RUN=true` in `.env` is enough — see "DRY_RUN" below.

Tests (pytest; the dev extras bring pytest, httpx for FastAPI's TestClient and
scikit-image for `scripts/nrrd_to_stl.py`):

```bash
pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q test_exam.py test_upload_exam.py test_boolean.py test_obj_materials.py
```

From Claude Code, the workspace `.claude/launch.json` has `mesh-processor-dry`:
the same uvicorn in DRY_RUN with `DRY_RUN_GLB_DIR` pointing at the viewer root
and `VIEWER_BASE=http://127.0.0.1:5500/case/`, so a local upload opens locally.

### Testing the upload endpoint

Multi-file form (normal case — one clinical case has several structures):

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@artery.stl" \
  -F "files=@vein.stl" \
  -F "files=@kidney.stl"
```

The `name` form field **does not exist** — the case uid is minted here (`secrets.token_hex(16)`, 32 hex chars, the same shape the Sketchfab uids had). The original STL filenames are cleaned (see `clean_mesh_names` in `main.py`) and become the GLB node names that the viewer displays as toggle labels.

### Building/running with Docker

```bash
docker build -t mesh-processor .
docker run -p 8000:8000 -e DRY_RUN=true mesh-processor
```

The Dockerfile installs `build-essential` because `fast-simplification` compiles a C++ extension on `pip install` and there's no prebuilt wheel for `linux/aarch64`.

### Deployment

Hosted on Railway **transitionally**. Push to `main` triggers auto-deploy. Railway auto-detects the Dockerfile and uses it. Required env vars (set in the Railway dashboard):
- `R2_ACCOUNT_ID` — Cloudflare account identifier (secret-ish; non-authenticating but unique to the account)
- `R2_ACCESS_KEY_ID` — R2 token Access Key (secret)
- `R2_SECRET_ACCESS_KEY` — R2 token Secret (secret)
- `R2_BUCKET` — defaults to `clinical-3d` if unset
- `VIEWER_BASE` — defaults to `https://biodesignlab.com.br/case/`
- `EXAM_WRITE_SECRET` — signs the `write_token` that lets the upload page add exam series 1..3 to a case (secret; any long random string — rotating it only breaks uploads in progress). Set it via Railway's Raw Editor.
- `PORT` — set automatically by Railway
- `DRY_RUN` — leave unset (or `false`) in production. Boot fails loudly if any of the R2 secrets or `EXAM_WRITE_SECRET` are missing while DRY_RUN is off, by design. (`SKETCHFAB_TOKEN` is no longer read since Sprint 3c.)

**Future host: Google Cloud Run** when migration triggers fire (LGPD pressure, GPU need for Sprint 4+, or cost crossover). Because this service runs entirely from `Dockerfile` with env vars at the edges, the migration is primarily learning `gcloud` CLI and re-setting env vars in the target dashboard. See the workspace-level `CLAUDE.md` for the full hosting strategy and triggers.

## Architecture

### Project structure

```
mesh-processor/
├── main.py          # FastAPI app — /upload, /cases/{uid}/exam, /health. Pure orchestration.
├── processor.py     # STL(s) → scene → multi-mesh GLB (pure, testable, no I/O)
├── r2.py            # Cloudflare R2 client (S3-compatible via boto3). Thin. DRY_RUN short-circuit inside.
├── exam.py          # Image exam (DICOM series / zip / NRRD) → canonical NRRD (pure, no I/O)
├── test_exam.py, test_upload_exam.py  # pytest: exam.py and the `exam` field end to end
├── scripts/nrrd_to_stl.py  # dev only: marching-cubes STL from a NRRD + the sphere test pair (and its 2nd series)
├── scripts/upload_fixtures.py  # dev only: the multi-series DICOM fixtures of medCaseViewer/tests/upload
├── test_processor.py  # Informal smoke test against real STLs (not pytest)
├── .env.example     # Template for VIEWER_BASE, R2_*, EXAM_WRITE_SECRET, DRY_RUN
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

### Critical separation: processor vs destination

`processor.py` knows nothing about R2. `r2.py` knows nothing about meshes. This is intentional, and it has paid off twice: Sprint 2 added `r2.py` as a sibling to `sketchfab.py` with zero processor changes, and Sprint 3c deleted `sketchfab.py` touching only `main.py`'s orchestration. **Do not couple them.**

For the same reason `processor.py` returns `bytes`, not a file path: where the bytes go is the caller's business. Keeping intermediates in memory costs ~10MB per request.

### Pure vs effectful split

`processor.py` is a **pure** module — `bytes in → bytes out`, no I/O, no env vars, no network. Call it 1000 times with the same input and it returns the same output. Testable without any setup.

`r2.py` is **effectful** — it mutates the world (writes R2 objects, consumes ops). That's why it has a DRY_RUN short-circuit: we want to exercise every layer above it (endpoint, CORS, the upload page, orchestration in `main.py`) without actually triggering effects until we're ready. (`sketchfab.py` followed the same rule until Sprint 3c deleted it.)

The rule: separate computation from side effects at file boundaries. Testing the pure part is free; testing the effectful parts costs dollars/time and should be done sparingly.

### Mesh processing decisions (medical context)

These defaults exist because this is **medical/surgical data**, not generic 3D content:

- **Target triangle count: 300,000 per mesh** (not per scene). Preserves anatomical detail (fractures, calcifications, vessel branches) while keeping each structure lean. Configurable per request via `target_triangles` form field.
- **Decimation algorithm: `fast_simplification` (quadric edge collapse).** Chosen over `pymeshlab` (heavy install, GPL) and `trimesh.simplify_quadric_decimation` (slower, less stable on large meshes).
- **No aggressive smoothing.** `trimesh.load(process=True)` does safe cleanup (duplicate vertices, normals). Anything more (Laplacian smoothing, Taubin) can round off clinically relevant features and is **off by default**. If a future request needs it, gate it behind an explicit flag, not a default.
- **Coordinate system: STL is patient space, Z-up (LPS, the 3D Slicer default); glTF is Y-up.** We apply a fixed `-π/2` rotation around X so each mesh lands upright in the viewer (`_RAS_TO_GLTF` — the name is historical; no x/y sign flip happens, so the GLB is **LPS rotated**). The rotation is identical for every mesh in a batch, preserving inter-structure spatial relationships (a kidney, its artery, its vein, and a lesion from the same exam stay co-registered). An STL whose 80-byte header says `SPACE=RAS` (Slicer writes `SPACE=LPS` or `SPACE=RAS` there) is first rotated 180° about S (`diag(-1,-1,1)`) to LPS. The viewer applies the same rotation to the image exam (`medCaseViewer/case/exam-geom.js`) — **change both or neither**.
- **Output format: GLB binary.** Smaller than glTF+bin, single file, native browser support; Three.js `GLTFLoader` reads it directly.
- **Per-mesh PBR materials, not vertex colors.** Each structure gets a named `PBRMaterial` (`baseColorFactor` + `roughnessFactor=0.5` + `metallicFactor=0`). Reason: the viewer's per-structure opacity and color work on each structure's material (the legacy Sketchfab viewer's `api.setMaterial` also worked on the *material list*). Without distinct materials, the opacity slider per structure cannot function. Vertex colors would render visually but would be a single material in the viewer.

### Color assignment (keyword-based with colorblind-safe fallback)

`main.py > clean_mesh_names` strips common prefix/suffix across the batch plus a `_(timestamp)` regex, then transliterates accents. The resulting lowercased name is matched against keywords in `processor.COLORS_BY_KEYWORD`:

| Keyword substring | Color | Meaning |
|---|---|---|
| `art` | `#BD0006` | artéria (dark red) |
| `vei` | `#477EFF` | veia / vein (blue) |
| `rim` | `#BA5531` | rim (brown-orange) |
| `lesao` | `#08E700` | lesão (bright green) |
| `tumor` | `#08E700` | tumor (same green as lesão — shares bucket, see below) |
| `pele` | `#FFD09C` | pele (skin tone) |
| `cortex` | `#966830` | córtex (brown) |

The table above is illustrative — `COLORS_BY_KEYWORD` in `processor.py` is the source of truth (it also has `rins`/`kidney`/`renal`, `osso`, and the order rules).

**English names (since 2026-10-02).** Automatic segmentations (TotalSegmentator, 3D Slicer) export English names, which used to fall into the fallback palette (a case arrived with blue `Bones` and a blue `stomach`). Each organ now has its Portuguese and English keyword on the same hex: `lesion`, `skin`, `bone`, and the abdomen organs `figado`/`liver`, `baco`/`spleen`, `estomago`/`stomach`, `duoden`, `esofag`/`esophag`. Names are **not** translated — the viewer shows the name the file came with.

**The hex is written raw into `baseColorFactor`, which glTF reads as linear**, so on screen a color looks lighter than its hex (`#EAE3D2` bone shows as `#F6F2EA`). For the abdomen organs the color was chosen by what the viewer shows, so their hexes are dark (liver `#450B06` shows as `#8E3A2A`); the on-screen color is noted next to each entry. Dark bases sit near `_vary_hsv`'s V floor (0.30), so duplicates of liver/spleen differ less than duplicates of the bright colors.

Non-matched names cycle through an IBM Colorblind Safe palette (`FALLBACK_COLORS`) by index — deterministic, so the same name consistently gets the same color. To add/change a clinical category, edit `COLORS_BY_KEYWORD`; don't touch the fallback palette lightly — reordering it reshuffles colors for unmatched cases.

**Duplicates share a "color bucket" and get HSV-varied.** When two or more meshes resolve to the same base hex (two `art`-prefixed structures, or `lesao` + `tumor` in the same case, or fallback palette wrapping past index 4), `_vary_hsv` walks each duplicate through progressively darker V with alternating S offsets. The first occurrence keeps the base color; subsequent ones stay visually distinguishable in the viewer's toggle list without losing the clinical meaning of the hue. Bucketing is per-hex, not per-keyword, which is why `tumor` mapping to the same green as `lesao` works correctly out of the box.

**Metal finish (special case).** A structure whose name contains `metal` (implant, screw, plate, stent) is the one case where the PBR *finish* changes, not just the color: it gets `metallicFactor=1.0` + low `roughnessFactor` (see `METAL_*` constants in `processor.py`) and a neutral steel/titanium base hex, so the viewer's environment map renders it as polished metal. Every other structure keeps the fixed `metallic=0 / roughness=0.5`. `metal` counts as a keyword match (no fallback slot consumed) and its base hex still flows through the duplicate-bucket / HSV logic, so two distinct metal parts in one case stay distinguishable.

### Splitting one structure by another: dentro/fora (STL path only)

The upload page can pair a **referência** (A) with a structure **a dividir**
(B), sent in the optional `boolean_ops` form field as JSON
`[{"principal": "<filename>", "secondary": "<filename>"}]` (original filenames
of the same request; `main._parse_boolean_ops` validates and maps them to
cleaned names). The field and key names are **frozen API contract**; all
user-facing wording is clinical (never "booleana" / "interseção" — radiologists
think anatomy, not set operations). Fixed product semantics, per pair, in order:

- **A stays whole** — untouched.
- **B is renamed `B fora de A`** and carries the geometry B − A. If that is
  empty (B fully inside A), B is dropped entirely.
- A new mesh **`B dentro de A`** (B ∩ A) is inserted right after B. If empty
  (structures don't overlap), the whole request fails 400: a misconfiguration
  the clinician must fix.

Those names ARE the node names the viewer displays, so the upload preview chip
and the viewer's structure list read identically.

Implementation: `processor._apply_boolean_ops`, using `trimesh.boolean` with
the **manifold3d** engine. Runs **after decimation** (smaller meshes → faster
operation, lean output) and **before** the RAS→glTF rotation and coloring. Both
operands must be watertight (`is_watertight` pre-check with a pt-BR error); the
internal index is keyed by the **original** names, so chained splits work after
the rename and their labels compose (`Tumor fora de Rim dentro de Coluna`).

**`_clean_boolean_result` is required, not cosmetic.** manifold3d returns a few
zero-area faces; with them trimesh counts edges shared by more than two faces
and reports the result as NOT watertight (`euler_number` 6 instead of 2). That
broke chained splits at the watertight pre-check and would leave "open" pieces
that the viewer's volume mode flags with `~`. `process(validate=True)` clears it
without changing the volume.

Colors: both pieces are colored from the **origin structure, never from the
composed name** — `Veia fora de Tumor` contains "tumor" and would match the
green. `_LoadedMesh.color_name` keeps the name the structure arrived with, and
the outer piece (`B fora de A`) is colored by it, so it keeps B's own color. The
inner piece (`B dentro de A`) points to B via `isolated_from` and gets a
**lighter tone of B's final color** (`_isolated_piece_material`: 30% of the way
to white; origins that are already near-white — bone, metal, HLS L > 0.72 —
go 35% toward black instead). Same hue keeps the anatomy readable (the vein
stays blue); the lightness step makes it visible that there are now two
structures. Lighter, not darker, because darker already means "another
structure of the same color" (`_vary_hsv`). Two inner pieces from the same
origin (chaining) share a tint bucket and get HSV-varied. The inner piece is
always matte, even from a `metal` origin (the outer piece stays metallic).
The upload screen does not know these colors — it shows neutral bars. STL-only:
OBJ bundles reject `boolean_ops` with a 400.

### Image exam: the `exam` field and extra series (DICOM series or NRRD → canonical NRRD)

A case can also carry the imaging exam it was segmented from — up to 4 series
(e.g. arterial and nephrographic phases); the viewer shows axial/coronal/sagittal
slices, the planes inside the 3D scene, and a series switcher. A case may be
structures only, exam only, or both.

- **One series per request** (Railway gives a request body 5 minutes; 4 thin CT
  phases are ~1 GB). Series 0 — the one the structures were segmented on — comes
  in `POST /upload`; series 1..3 in `POST /cases/{uid}/exam`.
- **`POST /upload`:** `exam` is a repeated multipart field like `files` (frozen
  name): N DICOM files (extension irrelevant — sniffed by the `DICM` preamble;
  raw datasets without preamble/file meta are read with `force` and get their
  Transfer Syntax from the encoding they were read with, `_ensure_transfer_syntax`),
  **or** one `.zip` of the series (what the upload page always sends for DICOM:
  a zip built in the browser with only the chosen series), **or** one `.nrrd`.
  Mixing → 400. `files` is optional; the request needs `files` or `exam`. When
  series 0 is stored, the response carries `write_token` =
  HMAC-SHA256(`EXAM_WRITE_SECRET`, uid) — stateless, no DB.
- **`POST /cases/{uid}/exam`:** form `write_token`, `index` (1..3), `exam` (one
  series, same normalization and limits). Wrong token or malformed uid → 403;
  `index` out of range → 400; R2 failure → 502. **Idempotent**: the same index
  overwrites — that is the upload page's "Tentar de novo". The server does not
  check that the series aligns with the structures (FrameOfReferenceUID): the
  page does it before sending; API clients are trusted.
- **Limits** (`main.py`), per request: `MAX_EXAM_BYTES` 200 MB (separate from
  the 60 MB mesh cap; fits the 5-min window at ~8 Mbps), `MAX_EXAM_FILES` 900
  (Starlette's multipart parser refuses >1000 files with an English message; we
  ask for the `.zip` in Portuguese first); `MAX_EXAM_SERIES` 4 per case. Peak
  memory measured on 2026-09-26: ~630 MB for a 500 MB (unzipped) series, ~3 s
  (`exam._read_dicom` keeps the series' datasets until stacking) — the Railway
  plan needs ≥ 1 GB for the service.
- **`exam.py` (pure):** reads one file at a time (keeps only position + pixels),
  groups by `SeriesInstanceUID`, drops series with < 8 images (localizer,
  scout, dose report) and images whose `ImageType` contains `LOCALIZER` (the
  plane-reference image Siemens stores *inside* MPR/MIP reformat series, in
  another orientation — without this a coronal reformat was refused as "mixed
  orientation"; found on a public TCIA CT, TCGA-CW-5590), refuses more than one
  remaining series (400 listing
  them — never "pick the biggest": the wrong phase would still look plausible),
  refuses mixed orientation, gaps/missing images, duplicate positions and
  multi-frame (logged). Sorts by position along the normal (never
  InstanceNumber). Geometry: `i = IOP[0:3]·PixelSpacing[1]`,
  `j = IOP[3:6]·PixelSpacing[0]`, `k = (IPP_last − IPP_first)/(n−1)`,
  origin = first IPP. Modality LUT in int32; output `int16` if it fits, else
  `uint16`, else `float32`. MONOCHROME1 is inverted. Above `MAX_VOXELS`
  (16 M ≈ 32 MB in the phone's memory) it resamples by **area mean with a
  spacing floor** (`_plan_resample` + `_Resampler`, same code for the DICOM
  path — slice by slice while decoding — and the NRRD path): find the smallest
  spacing s that fits, bring only the axes finer than s to s (factors need not
  be integers; box weights, rows sum to 1), leave coarser axes alone. Tied
  axes shrink together, so the in-plane pixel stays square — the old rule
  halved only the first of two equal axes and left a real CT at 1.48 × 0.74 mm.
  The new voxel b sits at b·f + (f−1)/2 in old index units, so the origin
  moves (f−1)/2 old voxels; `ExamStats.downsample` is the float factor per axis. `ExamStats.label`
  = the SeriesDescription through `clean_label` (control chars and `\0`
  padding out, spaces collapsed, ≤ 64 chars; `None` if empty) or, for a NRRD,
  the file name without extension.
- **Canonical output, per series n:** `cases/{uid}.exam-{n}.nrrd` — gzip NRRD
  with only `type, dimension, space: left-posterior-superior, sizes, space
  directions, kinds, endian, encoding, space origin` — and then
  `cases/{uid}.exam-{n}.json` = `{version: 1, label, images, shape, spacing,
  bytes}` (no `modality`: the label is the only header field made public) (`_store_exam_series`: NRRD first; the JSON is what makes the
  series exist for the viewer, so a NRRD left without JSON by a failure between
  the two puts is invisible and gets overwritten on retry). NRRD input in
  RAS/LAS is converted to LPS; every custom key/value is dropped. The whole
  DICOM header is left behind — **anonymization by construction** — **except
  the series label** (user decision, 2026-09-26: the SeriesDescription as it
  came, only cleaned). The pixels themselves (a face in a head CT) still need
  access control — Sprint 4. Array order is `(k, j, i)` in C order, written
  with `index_order="C"` so `sizes` and `space directions` are both `i j k`;
  `test_exam.py` has the transposition test for this.
- **Order in `/upload`:** normalize the exam first (an input error → 400 before
  anything is published: no orphan GLB) → meshes → R2 GLB (if there are
  meshes; failure → 502, see R2 notes) → R2 series 0. With a model the series-0
  put is best-effort: `exam.stored: false` + `exam.error` and
  `write_token: null` (no extra series without series 0). Without a model the
  exam *is* the case: failure → 502.
- **Exam-only case:** no GLB; `stats: null`. Like every case, uid =
  `secrets.token_hex(16)` and `processing: false`.
- **Response** (`exam` in `/upload`; the whole body of `/cases/{uid}/exam` is
  `{uid, exam}`): `exam: null | {stored, index, label, source, shape, spacing,
  downsample, size_mb, ignored_files, modality, error?}`. The viewer does not
  use it — it finds the series by `GET cases/{uid}.exam-{n}.json`.
- **Endpoints are plain `def`** (FastAPI runs them in a threadpool): decoding a
  JPEG2000 series and gzipping takes seconds and would otherwise block the event
  loop (`/health`).
- **Decoders:** `pydicom` + `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`,
  `pylibjpeg-rle` (compressed series from PACS; cp312 wheels, nothing to
  compile). No `python-gdcm`.
- **Future (asked by the user):** a segmentation NRRD (labelmap) becoming 3D
  structures via marching cubes. Today NRRD/DICOM is image only.

### Required: force vertex-normal compute after transforms

`apply_transform` invalidates trimesh's cached normals. The GLB exporter only writes the `NORMAL` attribute if normals exist on the mesh at export time. Without `NORMAL`, viewers render flat-shaded (visible triangle facets).

**The fix in `process_stls`:** after `mesh.apply_transform(...)`, access `_ = mesh.vertex_normals` to force lazy recompute. This single line is what makes the viewer render smooth. `trimesh.Trimesh(..., process=True)` handles the post-decimation case; the access handles the pass-through case. Both are needed.

`scipy` is a transitive dep of trimesh for this compute — it's pinned in `requirements.txt`. Without scipy, the `vertex_normals` access raises a swallowed `ModuleNotFoundError` and the GLB comes out without normals.

### Input limits

- **Image exam: per request 200 MB / 900 loose files / one series; 4 series per case; 16 M voxels per series after downsampling.** See "Image exam" above.
- **Max STL size: 60MB.** Set in `main.py`. STLs larger than this should not exist in our pipeline (segmentation output is capped upstream). If they do, fail loudly — silently truncating clinical data is dangerous.
- **STL only for now.** OBJ/PLY/etc. could be added but are not in scope.

### Sketchfab (removed in Sprint 3c)

Until 2026-09-27 every model was also uploaded to Sketchfab (`sketchfab.py`, `isDownloadable=true` to stay out of the Basic plan's 10/month cap) and the upload page polled `GET /status/{uid}` until Sketchfab finished processing. Both are gone: nothing in this service reads `SKETCHFAB_TOKEN` or calls the Sketchfab API. The ~100 cases created before Sprint 2 live only on Sketchfab (several on collaborators' accounts); the viewer opens them read-only through `/case/legacy/`, and the account stays alive at $0, never written to. The client and its field names (confirmed by a real upload on 2026-04-23) are in git history, before the Sprint 3c commit.

### Cloudflare R2 integration notes

R2 is the parallel storage backend added in Sprint 2. Why R2 (and not S3/GCS) lives in the workspace-level `CLAUDE.md` under "Hosting strategy" — short version: zero egress cost on Cloudflare's network, S3-compatible API so the SDK is just `boto3` pointed at a different `endpoint_url`, and a generous always-free tier (10GB storage, 1M Class A ops/month, 10M Class B ops/month).

**Object key convention:** `cases/{uid}.glb`, uid = `secrets.token_hex(16)` minted by `main.py` (until Sprint 3c it was the Sketchfab uid — also 32 hex chars, so the viewer URL contract did not change). The `cases/` prefix exists so the bucket can later host non-case objects (thumbnails, JSON metadata, exports) without name collisions.

**Failure semantics: the GLB put is mandatory.** R2 is the only copy of the model since Sprint 3c, so an `R2Error` on `cases/{uid}.glb` logs `[r2 ERROR] ...` and the request fails with 502 "Não foi possível guardar o modelo agora. Tente enviar de novo." — a 200 would hand the clinician a link that opens "caso não encontrado". (In Sprint 2 this was best-effort, because Sketchfab held the copy the viewer used.) The exam's series 0 is still best-effort when the case has a model — see "Image exam".

**Token scope (least privilege).** The R2 token used in production has permission "Object Read & Write" scoped to the single bucket `clinical-3d`. It cannot list, create, or delete buckets, and cannot touch other buckets in the account. If the token leaks, the blast radius is bounded to objects within `clinical-3d`.

**Endpoint URL is derived, not configured.** `r2.py` builds `https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com` from the account id. This is the default (auto/global) R2 endpoint — works because we picked "Localização: Automática" at bucket creation time. If a future bucket uses jurisdictional restriction (EU, FedRAMP), the endpoint format changes and this assumption breaks.

**`region_name="auto"`.** R2 has no real regions, but `boto3` requires `region_name` to construct request signatures. Cloudflare accepts `"auto"` (canonical) or any AWS-region-like string. Don't change this without a reason.

**`ContentType="model/gltf-binary"`.** Set on every put. The viewer fetches `cases/{uid}.glb` over HTTP, and the browser uses Content-Type to route the bytes to the right loader.

**No retries in `r2.py`.** Thin-client rule: HTTP + response parsing, nothing else. boto3's default retry behavior is left in place (it handles transient 503s with backoff internally), but we add no retry loops of our own. If we later observe lots of transient failures and want explicit retry policy, add it in `main.py` orchestration, not in the client.

### CORS

Only `https://biodesignlab.com.br` and the four local Live Server variants (`http://127.0.0.1:5500`, `http://localhost:5500`, `http://127.0.0.1:5501`, `http://localhost:5501`) are allowed. VSCode's Live Server defaults to `127.0.0.1:5500` — `127.0.0.1` and `localhost` are distinct origins to the browser, so both must be listed. Add new origins explicitly; do not use `*`.

## Code patterns

- **FastAPI with type hints.** Use `Form()`, `UploadFile`, and Pydantic models for request validation. Return plain dicts for responses (FastAPI handles serialization).
- **Errors as HTTPException with clear messages.** The frontend surfaces these directly to clinicians, so keep them human-readable in Portuguese where user-facing. Ex: `raise HTTPException(400, "STL inválido ou corrompido: ...")`.
- **No background workers / queues yet.** Processing happens synchronously in the request. Multi-STL cases process in <1s plus the R2 put. If we add larger inputs or batch processing, revisit with Celery + Redis.
- **Stateless service.** No database. R2 keys are the only state (`cases/{uid}.*`). Still no DB until metadata requirements appear.
- **Pure functions in `processor.py`.** Takes bytes, returns bytes + stats. No I/O, no env vars. Makes it trivially testable and reusable.
- **Thin clients** (`r2.py`; `sketchfab.py` had the same shape until Sprint 3c). Only HTTP + response parsing. No retry loops, no caching, no polling. Orchestration (retry, concurrency, cross-destination logic, failure tolerance) lives in `main.py`. That is what made Sprint 3c cheap: deleting `sketchfab.py` touched only `main.py`.

### DRY_RUN pattern

The `DRY_RUN=true` env var short-circuits the effectful client:
- `r2.upload_glb` / `r2.upload_exam` / `r2.upload_exam_meta` log `[r2 DRY_RUN] upload '<key>' (<bytes>) -> bucket=<bucket>` and return without calling the R2 API. With `DRY_RUN_GLB_DIR` set (name kept from when only the GLB existed) the objects are written to `<dir>/cases/{uid}.glb` and `<dir>/cases/{uid}.exam-{n}.nrrd|.json`, which the local viewer opens before trying R2 (the uid is random in DRY_RUN too, so without `DRY_RUN_GLB_DIR` the returned `viewer_url` opens "caso não encontrado"; the workspace launch config `mesh-processor-dry` sets it to the viewer root).
- `EXAM_WRITE_SECRET` falls back to a fixed dev-only value, so write tokens work locally without configuration.

That's enough to exercise the upload page, CORS, and the full UX loop end-to-end without spending R2 ops.

`DRY_RUN=true` also lets the service boot without **any** of `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` or `EXAM_WRITE_SECRET` — useful for CI and for hand-off to new contributors who don't have credentials yet. Production must never set `DRY_RUN=true`.

The fake/real paths living in the same file is deliberate: if a real API's shape changes (R2 endpoint URL convention, etc.), the DRY_RUN branch is right there, a few lines away, and gets updated in the same diff. A separate mock file would drift silently.

### Error handling inside `processor.py`

`_load_and_decimate` catches **any** exception from `trimesh.load` and re-raises as `ValueError("STL inválido ou corrompido: ...")`. Reason: trimesh raises a variety of exception types depending on what's wrong with the STL (ModuleNotFoundError for missing optional deps, KeyError for malformed internal refs, etc.). The contract guaranteed by `processor.py` to `main.py` is: "bad input → ValueError, always." This lets `main.py` catch exactly that and return HTTP 400 without guessing.

Transitive deps worth knowing about:
- `scipy` — needed by trimesh to compute per-vertex normals (see smooth-shading note above).
- `chardet` — needed by trimesh's STL ASCII fallback when the binary parse fails. Without it, feeding random bytes to `trimesh.load` raises `ModuleNotFoundError` (silently without our `except`).

## Roadmap (do not break these paths)

- **Sprint 1 ✅:** STL → optimized GLB → Sketchfab. Viewer URL pattern unchanged.
- **Sprint 2 ✅:** `r2.py` added. After Sketchfab upload succeeds, also push GLB to Cloudflare R2 at `cases/{uid}.glb` (best-effort — R2 failure does not break the request). Same UID, two storage backends.
- **Sprint 3 ✅:** `medCaseViewer/case/` rewritten from the Sketchfab iframe to Three.js + GLTFLoader reading from R2, with the Sketchfab iframe kept as read-only fallback for pre-Sprint-2 cases (`/case/legacy/`). **3c (2026-09-27):** Sketchfab removed from this service — R2 is the only destination, uids are minted here, `/status` is gone. Public URL (`?id=...`) stays identical — clinicians with old links keep working.
- **Sprint 3d ✅:** image exam (DICOM/NRRD → canonical NRRD in R2, up to 4 series per case) — see "Image exam".
- **Future (after AI pipeline):** A separate `ai-segmentation` service will produce STLs from DICOM and POST them to this service's `/upload` endpoint. This service does not need to know whether the STL came from a human upload or an AI run.

The canonical, cross-service version of this roadmap lives in the workspace-level `CLAUDE.md` (one folder up). If the two ever disagree, that one wins.

## What this service is NOT responsible for

- AI segmentation (separate service, not built yet). DICOM *normalization* to the canonical NRRD does live here (`exam.py`), because it is storage preparation for the viewer, not analysis.
- Authentication or user accounts (out of scope; unguessable case UIDs are the access control for now).
- Storing case metadata (patient name, exam date, etc.) — when this is needed, add a Postgres on Railway and a separate `cases` service. Do not bolt it onto this one.
- Frontend rendering / Three.js / measurement tools (lives in the `medCaseViewer` repo).
