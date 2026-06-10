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

## Splatting to Mesh

The **Splatting > 3DGS Meshization** Library item creates one Empty scene object with:

- `SplatMeshConversion`: 3DGS `.ply` input, density-grid mesh extraction settings, output status, and the `Convert 3DGS to Mesh` solver button.

Meshization reads a 3DGS PLY inside the opened Project and writes a triangle mesh PLY under:

```text
.witwin/runtime/wt_splatting/mesh/<run_id>/
```

The built-in `density` method builds a Gaussian density field, runs Marching Cubes, and colors mesh vertices from nearby splats. This is intended as a quick collision/proxy/inspection mesh, not a production retopology pass.

The `sugar` and `2dgs` methods are external backend adapters. They keep third-party CUDA repositories out of the plugin source while making Studio, solver, and CLI execution use the same interface. Point the component or CLI at an official backend repo, or configure a custom adapter script. For `2dgs`, an image directory is accepted directly: the plugin stages the images, runs COLMAP, normalizes the COLMAP output into the `sparse/0 + images` layout expected by 2DGS, trains 2DGS, then runs mesh extraction.

```powershell
$env:WT_SPLATTING_SUGAR_REPO = "E:\path\to\SuGaR"
$env:WT_SPLATTING_2DGS_REPO = "E:\path\to\2d-gaussian-splatting"
$env:WT_SPLATTING_SUGAR_SCRIPT = "E:\path\to\sugar_adapter.py"
$env:WT_SPLATTING_2DGS_SCRIPT = "E:\path\to\2dgs_adapter.py"
```

Adapter scripts are called with:

```text
--input <scene-or-splat-path> --output-dir <run-dir> --mesh-output <run-dir>/<output_name>.ply --iterations <steps> [--repo <backend_repo>]
```

They must write the requested mesh PLY path or another `.ply` under the output directory.
The built-in 2DGS adapter also accepts `--mesh-res` and `--mesh-cluster-count`; the component and benchmark CLI expose these as `mesh_resolution` and `mesh_cluster_count`. Use a multiple of 512 for the official unbounded 2DGS mesh extractor.

## Prerequisites

- COLMAP installed and available as `colmap`, or set the component `colmap_binary` field to an executable path.
- PyTorch with CUDA for practical training. CPU mode is intended for small debugging runs only.
- `gsplat` installed in the solver Python environment.
- `scikit-image` installed in the solver Python environment for 3DGS-to-mesh extraction.
- `huggingface_hub` installed for the Mip-NeRF 360 benchmark downloader.
- A gsplat training adapter configured through one of:
  - `WT_SPLATTING_GSPLAT_ADAPTER=module:function`
  - `WT_SPLATTING_GSPLAT_TRAIN_SCRIPT=C:\path\to\train.py`
- SuGaR and 2DGS official repos or adapter scripts configured through:
  - `WT_SPLATTING_SUGAR_REPO=C:\path\to\SuGaR`
  - `WT_SPLATTING_2DGS_REPO=C:\path\to\2d-gaussian-splatting`
  - `WT_SPLATTING_SUGAR_SCRIPT=C:\path\to\sugar_adapter.py`
  - `WT_SPLATTING_2DGS_SCRIPT=C:\path\to\2dgs_adapter.py`

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
- external SuGaR and 2DGS mesh backends through adapter scripts
- Mip-NeRF 360 kitchen/room benchmark harness
- in-Studio preview through Spark

V1 does not support:

- video decoding
- COLMAP-free pose estimation
- cloud or off-machine training management
- built-in model cleanup or compression
- vendored SuGaR or 2DGS source

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

Run the full Mip-NeRF 360 SuGaR/2DGS benchmark with `witwin2`:

```powershell
& C:\Users\Asixa\miniconda3\envs\witwin2\python.exe plugins\wt_splatting\tools\benchmark_mipnerf360.py `
  --project E:\path\to\project `
  --methods sugar,2dgs `
  --scenes kitchen,room `
  --full-convergence `
  --backend-python C:\Users\Asixa\miniconda3\envs\witwin2\python.exe `
  --colmap-binary C:\Users\Asixa\miniconda3\envs\witwin2\Library\bin\colmap.exe `
  --mesh-resolution 1024 `
  --mesh-cluster-count 1 `
  --sugar-repo E:\path\to\SuGaR `
  --twodgs-repo E:\path\to\2d-gaussian-splatting
```

If `--dataset-root` is omitted, the tool uses `huggingface_hub.snapshot_download` for `nvs-bench/mipnerf360`. Results are written to:

```text
.witwin/runtime/wt_splatting/benchmarks/mipnerf360/
```
