This repository includes code derived from third-party open-source projects.

## TOAD (AGPL-3.0)
- Upstream: https://github.com/mahmoodlab/TOAD
- License: AGPL-3.0 (see LICENSE and licenses/TOAD_AGPL-3.0.txt)
- Notes: This repository contains a modified version of TOAD for tumor site classification.
- Included components: all scripts.
- Modifications in this repository:
  - primary-site classification only (architecture/loss adapted)
  - patient-level stratification enabled by default
  - additional logging and reproducibility utilities
  - containerized execution support
  - release of pretrained weights
  - separate test set with OOD mpp slides

## CLAM (GPL-3.0)
- Upstream: https://github.com/mahmoodlab/CLAM
- License: GPL-3.0 (see licenses/CLAM_GPL-3.0.txt)
- Notes: This repository includes a subset of CLAM code used for preprocessing.
- Included components: only scripts for WSI patching and feature extraction. Exclusion of the classifier training scripts.
- Modifications in this repository:

### 1) Changes in `wsi_core/wsi_utils.py` (utilities / module-level)
- **Added optional dependency handling** so the code can run even when some slide backends are missing:
  - `openslide` (SVS), `tifffile` (BTF/TIFF), `pyvips` (fast random-access for TIFF/BTF).
- **Introduced harmonization/QA helper utilities** used by `WholeSlideImage`:
  - basic font + text wrapping helpers for report images
  - functions to overlay a **10 µm grid**, draw **scale bars**, and compute a **similarity score** (SSIM if available; fallback otherwise)
- **Added OME-XML parsing logic (for BTF/TIFF)** to recover **PhysicalSizeX/Y (MPP)** and optional objective info.
  - Includes a selection policy to avoid grabbing low-res “overview” pages by mistake (prefers the main IFD/page).

---

### 2) Changes in `WholeSlideImage` (core class behavior)

#### A) Slide IO now supports SVS *and* BTF/TIFF
- Replaced direct OpenSlide-only access with a backend abstraction:
  - **SVS** → OpenSlide backend
  - **BTF/TIFF** → tifffile/pyvips backend
- `read_region()` is now proxied through the backend, keeping the rest of the pipeline compatible.

#### B) Harmonization: explicit, MPP-based scaling (main focus)
- `dx_target = target_mpp / mpp_x`, `dy_target = target_mpp / mpp_y`
- **Hard guard:** if MPP metadata is missing (`mpp_x` or `mpp_y`), harmonization is skipped and the slide is flagged as skipped.

#### C) Virtual segmentation pyramid for single-level slides
- Segmentation and preview avoid L0 when possible:
  - If the slide has multiple levels → choose a nonzero level near the desired downsample.
  - If it’s single-level → synthesize a virtual low-res proxy image for segmentation/preview.
  - Contours found on the proxy are mapped back to L0 coordinates by multiplying by the proxy downsample.

#### D) Harmonization QA artifacts (before/after evidence)
- Added routines to write a small harmonization report bundle per slide:
  - cropped **before** (native MPP) patch
  - cropped **after** patch read at harmonized native size then resized to target
  - side-by-side visualization with **10 µm grid**, scale bars, and **similarity metric**
  - JSON report containing key harmonization parameters and selected seg-level strategy

---

## Notes
- This repository is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE.
- Copyright and license headers from upstream-derived files should be preserved.
