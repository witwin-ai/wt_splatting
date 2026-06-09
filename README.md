# WiTwin Splatting

WiTwin Splatting provides Spark-compatible Gaussian Splat render resources for WiTwin Studio.

## 3DGS Reconstruction

The **Splatting > 3DGS Reconstruction** Library item creates one Empty scene object with:

- `SplatReconstruction`: image sequence, COLMAP, training, export, status, and the `Reconstruct 3DGS` solver button.
- `GaussianSplatRenderer`: the render component that receives the completed Spark-compatible splat resource.

Input images must live inside the opened Project. A typical path is:

```text
assets/splatting/captures/desk/images
```

Runtime files are written under:

```text
.witwin/runtime/wt_splatting/recon/<run_id>/
```

The renderable `.ply` is registered through the binary asset service after reconstruction. Visible `results/` output should be created only by an explicit export flow.

## 3DGS to Mesh

The **Splatting > 3DGS Meshization** Library item creates one Empty scene object with:

- `SplatMeshConversion`: 3DGS `.ply` input, density-grid mesh extraction settings, output status, and the `Convert 3DGS to Mesh` solver button.

Meshization reads a 3DGS PLY inside the opened Project and writes a triangle mesh PLY under:

```text
.witwin/runtime/wt_splatting/mesh/<run_id>/
```

The V1 extractor builds a Gaussian density field, runs Marching Cubes, and colors mesh vertices from nearby splats. This is intended as a quick collision/proxy/inspection mesh, not a production retopology pass.

## Prerequisites

- COLMAP installed and available as `colmap`, or set the component `colmap_binary` field to an executable path.
- PyTorch with CUDA for practical training. CPU mode is intended for small debugging runs only.
- `gsplat` installed in the solver Python environment.
- `scikit-image` installed in the solver Python environment for 3DGS-to-mesh extraction.
- A gsplat training adapter configured through one of:
  - `WT_SPLATTING_GSPLAT_ADAPTER=module:function`
  - `WT_SPLATTING_GSPLAT_TRAIN_SCRIPT=C:\path\to\train.py`

Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.webp`.

Recommended capture style:

- Use still images from a single static scene.
- Keep exposure and focus stable.
- Capture with strong overlap between neighboring views.
- Avoid fast motion blur, heavy reflections, and large lighting changes.
- Use at least 20 images for useful quality; small smoke captures can be much smaller.

## V1 Scope

V1 supports:

- unordered or sequential image folders
- COLMAP-based camera estimation
- 3DGS `.ply` export
- density-field 3DGS `.ply` to triangle mesh `.ply` conversion
- in-Studio preview through Spark

V1 does not support:

- video decoding
- COLMAP-free pose estimation
- cloud or off-machine training management
- built-in model cleanup or compression
- 2DGS reconstruction

## Smoke Test

Run the mock test suite:

```powershell
python -m pytest plugins\wt_splatting\tests\test_reconstruction.py -v
```

Run a real reconstruction only when COLMAP, PyTorch, gsplat, a training adapter, and a small capture are configured:

```powershell
$env:WT_SPLATTING_ENABLE_REAL_RECON = "1"
$env:WT_SPLATTING_SMOKE_PROJECT = "E:\path\to\project"
$env:WT_SPLATTING_SMOKE_IMAGE_DIR = "assets\splatting\captures\demo\images"
python -m pytest plugins\wt_splatting\tests\test_reconstruction.py -m requires_colmap -v
python plugins\wt_splatting\tools\reconstruct_smoke.py --project $env:WT_SPLATTING_SMOKE_PROJECT --image-dir $env:WT_SPLATTING_SMOKE_IMAGE_DIR
```

Run meshization on an existing 3DGS PLY:

```powershell
python plugins\wt_splatting\tools\meshize_smoke.py --project $env:WT_SPLATTING_SMOKE_PROJECT --splat-path "E:\path\to\project\.witwin\runtime\wt_splatting\recon\run\reconstruction.ply"
```
