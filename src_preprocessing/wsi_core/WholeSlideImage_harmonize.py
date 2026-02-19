"""
WholeSlideImage — .SVS or .BTF slides Harmonization, Virtual Segmentation Pyramid in case of single level
"""

import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py
import json

from wsi_core.wsi_utils import (
    savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator,
    save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
)
from wsi_core.util_classes import (
    isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
)
from utils.file_utils import load_pkl, save_pkl

Image.MAX_IMAGE_PIXELS = 933120000

# ------- Optional deps -------
try:
    import openslide
    HAS_OPENSLIDE = True
except Exception:
    HAS_OPENSLIDE = False

try:
    import tifffile as tiff
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False

try:
    import pyvips
    HAS_PYVIPS = True
except Exception:
    HAS_PYVIPS = False


# ---------- Module-level helpers (do NOT indent inside the class) ----------

def _load_font(pt=16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, pt)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    words = text.split()
    if not words:
        return [""]
    lines, buf = [], words[0]
    for w in words[1:]:
        test = buf + " " + w
        if draw.textlength(test, font=font) <= max_w:
            buf = test
        else:
            lines.append(buf)
            buf = w
    lines.append(buf)
    return lines


def _grid_overlay_um(img_np, um_per_px=None, um_step=10.0, fallback_px=28, rects=(0, 24, 48)):
    """
    Draw a grid with ~um_step micrometer spacing. If um_per_px is None, use fallback_px pixels.
    """
    im = Image.fromarray(img_np)
    d = ImageDraw.Draw(im)
    W, H = im.size

    if um_per_px and um_per_px > 0:
        step_px = max(1, int(round(um_step / float(um_per_px))))
    else:
        step_px = int(fallback_px)

    for x in range(0, W, step_px):
        d.line([(x, 0), (x, H - 1)], fill=(255, 255, 255), width=1)
    for y in range(0, H, step_px):
        d.line([(0, y), (W - 1, y)], fill=(255, 255, 255), width=1)

    cx, cy = W // 2, H // 2
    for r in rects:
        x0, y0 = cx - (W // 2 - r), cy - (H // 2 - r)
        x1, y1 = cx + (W // 2 - r), cy + (H // 2 - r)
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    return np.array(im)


def _scale_bar(img_w, img_h, um_per_px, target_fraction=0.25, height_px=8, margin=12):
    import numpy as np
    target_um = um_per_px * img_w * target_fraction
    nice = [1, 2, 5]
    pow10 = 10 ** int(np.floor(np.log10(max(target_um, 1e-6))))
    best_um = min((n * pow10 for n in nice + [10]), key=lambda v: abs(v - target_um))
    px = max(1, int(round(best_um / um_per_px)))
    x1 = img_w - margin
    x0 = x1 - px
    y1 = img_h - margin
    y0 = y1 - height_px
    label = f"{int(best_um) if best_um >= 10 else round(best_um, 1)} µm"
    return x0, y0, x1, y1, label


def _compute_ssim_01(a_rgb, b_rgb):
    import numpy as np
    a = Image.fromarray(a_rgb).convert("L")
    b = Image.fromarray(b_rgb).convert("L")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    try:
        from skimage.metrics import structural_similarity as ssim
        s, _ = ssim(
            a, b, data_range=255, full=True,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False
        )
        return float(np.clip(s, 0.0, 1.0)), "ssim"
    except Exception:
        mse = np.mean((a - b) ** 2)
        nm = float(np.clip(1.0 - (mse / (255.0 ** 2)), 0.0, 1.0))
        return nm, "nmse_fallback"


# ============================ Backend Abstractions ============================

class _SVSBackend:
    def __init__(self, path):
        assert HAS_OPENSLIDE, "OpenSlide not available for SVS."
        self.slide = openslide.open_slide(path)
        self.path = path
        self.kind = "svs"
        self.level_dimensions = self.slide.level_dimensions
        self.level_count = self.slide.level_count
        downs = []
        base_w, base_h = self.level_dimensions[0]
        try:
            for i, dim in enumerate(self.level_dimensions):
                if i < len(self.slide.level_downsamples):
                    ds = float(self.slide.level_downsamples[i])
                    downs.append((ds, ds))
                else:
                    w, h = dim
                    iso = 0.5 * (base_w / float(w) + base_h / float(h))
                    downs.append((iso, iso))
        except Exception:
            for dim in self.level_dimensions:
                w, h = dim
                iso = 0.5 * (base_w / float(w) + base_h / float(h))
                downs.append((iso, iso))
        self.level_downsamples = downs

        # MPP/objective
        self.mpp_x = self._get_float_prop("openslide.mpp-x")
        self.mpp_y = self._get_float_prop("openslide.mpp-y")
        self.objective = self._get_float_prop("openslide.objective-power")

    def _get_float_prop(self, key):
        try:
            v = self.slide.properties.get(key)
            return float(v) if v is not None else None
        except Exception:
            return None

    def read_region(self, location, level, size):
        return self.slide.read_region(location, level, size)  # RGBA

    def get_downscaled_full(self, size, prefer_rgba=True):
        W, H = size
        thumb = self.slide.get_thumbnail((W, H))  # PIL RGB
        if thumb.size != (W, H):
            thumb = thumb.resize((W, H), Image.BICUBIC)
        if prefer_rgba and thumb.mode != "RGBA":
            thumb = thumb.convert("RGBA")
        return thumb


class _BTFBackend:
    OME_NS = {"OME": "http://www.openmicroscopy.org/Schemas/OME/2015-01"}

    def __init__(self, path):
        assert HAS_TIFFFILE, "tifffile not available for BTF/TIFF."
        self.path = path
        self.kind = "btf" if path.lower().endswith(".btf") else "tiff"
        self._vips = None
        self._tif = None
        self._zarr = None

        # --- Always decide the "main" IFD with tifffile (tiled + largest area) ---
        prefer_ifd = self._choose_main_ifd_index()

        # Prefer pyvips for data IO, but open it ON THE MAIN IFD/PAGE
        if HAS_PYVIPS:
            page_kw = {"page": int(prefer_ifd)} if prefer_ifd is not None else {}
            self._vips = pyvips.Image.new_from_file(self.path, access="random", **page_kw)
            try:
                libvips_version = ".".join(str(pyvips.version(i)) for i in (0, 1, 2))
            except TypeError:
                libvips_version = getattr(pyvips, "__version__", "unknown")
            print(
                f"[BTF] pyvips enabled: {pyvips.__version__}  "
                f"libvips={libvips_version}  access=random  page={page_kw.get('page', 0)}"
            )

            base_w, base_h = int(self._vips.width), int(self._vips.height)
            self.level_dimensions = [(base_w, base_h)]
            self.level_count = 1
            self.level_downsamples = [(1.0, 1.0)]
        else:
            print("[BTF] pyvips not available; using tifffile+zarr fallback")
            self._tif = tiff.TiffFile(self.path)
            series = self._tif.series[0]
            base = series.levels[0] if len(series.levels) else series
            base_h, base_w = base.shape[-2], base.shape[-1]
            self.level_dimensions = [(int(base_w), int(base_h))]
            self.level_count = 1
            self.level_downsamples = [(1.0, 1.0)]
            try:
                self._zarr = series.aszarr(chunkmode="page")
            except Exception:
                self._zarr = series.aszarr()

        # --- OME-XML via tifffile; prefer the IFD we chose above ---
        ome_xml = self._read_ome_xml()
        self.mpp_x, self.mpp_y, self.objective = self._parse_ome_for_base(
            ome_xml,
            base_wh=self.level_dimensions[0],   # now this is the high-res page dims
            prefer_ifd=prefer_ifd
        )

        if self.mpp_x is None or self.mpp_y is None:
            print("[BTF] WARN: OME-XML missing PhysicalSizeX/Y; MPP unavailable for harmonization.")

    def _choose_main_ifd_index(self):
        try:
            import tifffile as _tf
            best = None
            best_idx = None
            with _tf.TiffFile(self.path) as _t:
                for i, page in enumerate(_t.pages):
                    try:
                        w = int(page.tags.get("ImageWidth").value)
                        h = int(page.tags.get("ImageLength").value)
                    except Exception:
                        continue
                    tiled = 1 if getattr(page, "is_tiled", False) else 0
                    area = w * h
                    key = (tiled, area)
                    if best is None or key > best:
                        best, best_idx = key, i
            return best_idx
        except Exception:
            return None

    def __del__(self):
        try:
            if self._tif is not None:
                self._tif.close()
        except Exception:
            pass

    def _read_ome_xml(self):
        if self._tif is not None:
            for page in self._tif.pages:
                tg = page.tags.get("ImageDescription")
                if not tg:
                    continue
                val = tg.value if isinstance(tg.value, str) else tg.value.decode("utf-8", errors="ignore")
                if isinstance(val, str) and "<OME" in val:
                    return val
            return None
        try:
            with tiff.TiffFile(self.path) as tif:
                for page in tif.pages:
                    tg = page.tags.get("ImageDescription")
                    if not tg:
                        continue
                    val = tg.value if isinstance(tg.value, str) else tg.value.decode("utf-8", errors="ignore")
                    if isinstance(val, str) and "<OME" in val:
                        return val
        except Exception:
            pass
        return None

    def _parse_ome_for_base(self, image_description: str, base_wh, prefer_ifd=None):
        """
        Return (mpp_x, mpp_y, nominal_magnification) for the correct image.

        Selection policy (to avoid the 'overview' trap):
          1) If any OME:Image has <TiffData IFD="k"> matching prefer_ifd, use that.
          2) Else choose the image with the smallest PhysicalSizeX (highest-res),
             tie-break by largest SizeX*SizeY.
          3) Else (no PhysicalSizeX anywhere), choose largest SizeX*SizeY.
        Units normalized to µm.
        """
        if not isinstance(image_description, str) or "<OME" not in image_description:
            return None, None, None

        try:
            root = ET.fromstring(image_description)
        except Exception:
            cleaned = "".join(ch for ch in image_description if ch >= " " or ch in "\t\r\n")
            try:
                root = ET.fromstring(cleaned)
            except Exception:
                return None, None, None

        ns = self.OME_NS

        # Objective ID -> nominal magnification map
        obj_mag = {}
        for obj in root.findall(".//OME:Objective", ns):
            oid = obj.attrib.get("ID")
            mag = obj.attrib.get("NominalMagnification")
            if oid and mag:
                try:
                    obj_mag[oid] = float(mag)
                except Exception:
                    pass

        def _i(x):
            try:
                return int(x)
            except Exception:
                return None

        def _f(x):
            try:
                return float(x)
            except Exception:
                return None

        def _to_um(val, unit):
            if val is None:
                return None
            u = (unit or "µm").strip().lower()
            if u in ("µm", "um", "micrometer", "micrometre"):
                return val
            if u in ("nm", "nanometer", "nanometre"):
                return val / 1000.0
            if u in ("mm", "millimeter", "millimetre"):
                return val * 1000.0
            return val

        candidates = []
        for img in root.findall(".//OME:Image", ns):
            px = img.find(".//OME:Pixels", ns)
            if px is None:
                continue
            sx = _i(px.attrib.get("SizeX"))
            sy = _i(px.attrib.get("SizeY"))
            npix = (sx or 0) * (sy or 0)
            psx = _f(px.attrib.get("PhysicalSizeX"))
            psy = _f(px.attrib.get("PhysicalSizeY"))
            ux = px.attrib.get("PhysicalSizeXUnit") or px.attrib.get("PhysicalSizeUnitX")
            uy = px.attrib.get("PhysicalSizeYUnit") or px.attrib.get("PhysicalSizeUnitY")
            ifd_list = []
            for td in px.findall(".//OME:TiffData", ns):
                if "IFD" in td.attrib:
                    try:
                        ifd_list.append(int(td.attrib["IFD"]))
                    except Exception:
                        pass
            mag_val = None
            objset = img.find(".//OME:ObjectiveSettings", ns) or px.find(".//OME:ObjectiveSettings", ns)
            if objset is not None:
                ref = objset.attrib.get("ID") or objset.attrib.get("ObjectiveRef")
                if ref and ref in obj_mag:
                    mag_val = obj_mag[ref]
                if mag_val is None:
                    try:
                        mag_val = float(objset.attrib.get("NominalMagnification"))
                    except Exception:
                        pass

            candidates.append(
                dict(
                    sx=sx, sy=sy, npix=npix,
                    psx=psx, psy=psy, ux=ux, uy=uy,
                    ifds=ifd_list, mag=mag_val
                )
            )

        if not candidates:
            return None, None, None

        # 1) prefer IFD match
        chosen = None
        if prefer_ifd is not None:
            for c in candidates:
                if prefer_ifd in (c["ifds"] or []):
                    chosen = c
                    break

        # 2) smallest PhysicalSizeX (highest-res), tie by npix
        if chosen is None:
            with_ps = [c for c in candidates if c["psx"] is not None]
            if with_ps:
                with_ps.sort(key=lambda c: (c["psx"], -c["npix"]))
                chosen = with_ps[0]
            else:
                # 3) fallback: largest pixel count
                candidates.sort(key=lambda c: -c["npix"])
                chosen = candidates[0]

        mpp_x = _to_um(chosen["psx"], chosen["ux"])
        mpp_y = _to_um(chosen["psy"], chosen["uy"])

        mag = chosen["mag"]
        if mag is None and obj_mag:
            try:
                mag = max([m for m in obj_mag.values() if m is not None])
            except Exception:
                mag = None

        return mpp_x, mpp_y, mag

    def _extract_base_area(self, x0, y0, w0, h0):
        W, H = self.level_dimensions[0]
        if x0 < 0:
            w0 += x0
            x0 = 0
        if y0 < 0:
            h0 += y0
            y0 = 0
        w0 = max(0, min(int(w0), W - int(x0)))
        h0 = max(0, min(int(h0), H - int(y0)))
        if w0 <= 0 or h0 <= 0:
            return np.zeros((0, 0, 3), np.uint8)

        # Fast path: pyvips window read
        if self._vips is not None:
            region = self._vips.extract_area(int(x0), int(y0), int(w0), int(h0))
            mem = region.write_to_memory()
            arr = np.ndarray(
                buffer=mem,
                dtype=np.uint8,
                shape=(region.height, region.width, region.bands),
            )
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            elif arr.shape[2] > 4:
                arr = arr[:, :, :4]
            return arr

        # Zarr-backed slice (no reopen); only pulls necessary tiles
        if self._zarr is None:
            # ultra-fallback (shouldn’t happen with the init above)
            with tiff.TiffFile(self.path) as tif:
                series = tif.series[0]
                z = series.aszarr()
                sub = z[int(y0):int(y0 + h0), int(x0):int(x0 + w0)]
        else:
            sub = self._zarr[int(y0):int(y0 + h0), int(x0):int(x0 + w0)]

        arr = np.asarray(sub)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < 10:
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[2] > 4:
                arr = arr[:, :, :4]
        return arr.astype(np.uint8, copy=False)

    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        if level < len(self.level_downsamples):
            dsx, dsy = self.level_downsamples[level]
        else:
            dsx = dsy = float(2 ** level)
        x0 = int(round(x * dsx))
        y0 = int(round(y * dsy))
        w0 = int(round(w * dsx))
        h0 = int(round(h * dsy))
        base = self._extract_base_area(x0, y0, w0, h0)
        if base.size == 0:
            return Image.fromarray(np.zeros((h, w, 4), np.uint8), mode="RGBA")
        if base.ndim == 2:
            base = np.repeat(base[..., None], 3, axis=2)
        if base.shape[2] == 3:
            img = Image.fromarray(base, mode="RGB").convert("RGBA")
        else:
            img = Image.fromarray(base[:, :, :4], mode="RGBA")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BICUBIC)
        return img

    def get_downscaled_full(self, size, prefer_rgba=True):
        W, H = size
        if self._vips is not None:
            scale_x = W / float(self._vips.width)
            scale_y = H / float(self._vips.height)
            img = self._vips.resize(scale_x, vscale=scale_y)
            mem = img.write_to_memory()
            arr = np.ndarray(
                buffer=mem,
                dtype=np.uint8,
                shape=(img.height, img.width, img.bands),
            )
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            pil = Image.fromarray(arr[:, :, :3], mode="RGB")
            if prefer_rgba:
                pil = pil.convert("RGBA")
            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.BICUBIC)
            return pil

        # Zarr subsampling thumbnail (don’t read full image)
        W0, H0 = self.level_dimensions[0]
        step_x = max(1, int(np.floor(W0 / float(W))))
        step_y = max(1, int(np.floor(H0 / float(H))))
        if self._zarr is not None:
            sub = self._zarr[::step_y, ::step_x]
        else:
            with tiff.TiffFile(self.path) as tif:
                sub = tif.series[0].asarray()[::step_y, ::step_x]
        arr = np.asarray(sub)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        pil = Image.fromarray(arr[:, :, :3], mode="RGB")
        if prefer_rgba:
            pil = pil.convert("RGBA")
        return pil.resize((W, H), Image.BICUBIC)


# ============================ Main WSI class =================================

class WholeSlideImage(object):
    def __init__(self, path):
        print(f"[WSI] Opening: {path}")
        real_path = os.path.realpath(path)
        self.name = os.path.splitext(os.path.basename(real_path))[0]
        ext = os.path.splitext(real_path)[1].lower()
        if ext == ".svs":
            self.backend = _SVSBackend(real_path)
        elif ext in (".btf", ".tif", ".tiff"):
            self.backend = _BTFBackend(real_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        print(f"slide extension: {ext}")

        self.wsi = self  # proxy compatibility

        self.level_dim = self.backend.level_dimensions
        self.level_downsamples = self._assertLevelDownsamples()
        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

        self.mpp_x = self.backend.mpp_x
        self.mpp_y = self.backend.mpp_y
        self.objective = self.backend.objective

        # Aliases
        self.level_dimensions = self.level_dim
        self.level_count = len(self.level_dim)
        self.dimensions = self.level_dim[0]
        self.levels = self.level_dim
        self.properties = {
            "openslide.mpp-x": str(self.mpp_x) if self.mpp_x is not None else None,
            "openslide.mpp-y": str(self.mpp_y) if self.mpp_y is not None else None,
            "openslide.objective-power": str(self.objective) if self.objective is not None else None,
        }

        print("\n[INIT] =======================")
        print(f"[INIT] File     : {path}")
        print(f"[INIT] Kind     : {self.backend.kind}")
        print(f"[INIT] Dim L0   : {self.level_dim[0]}")
        print(f"[INIT] Levels   : {len(self.level_dim)}")
        print(f"[INIT] Downs    : {self.level_downsamples}")
        print(f"[INIT] MPP_X/Y  : {self.mpp_x} / {self.mpp_y}")
        print(f"[INIT] Obj.Power: {self.objective}")
        print("[INIT] =======================\n")

        # harmonization state
        self.harmonize = False
        self.target_mpp = None
        self.dx_target = None
        self.dy_target = None

        # segmentation mapping state
        self.seg_level = 0
        self.seg_down = 1.0
        self.seg_is_virtual = False
        self.dx_seg = None
        self.dy_seg = None

    # --------- OpenSlide-like proxy ---------
    def read_region(self, location, level, size):
        return self.backend.read_region(location, level, size)

    def getOpenSlide(self):
        return self

    # --------- helpers ---------
    def _closest_nonzero_level(self, target_downsample=32.0):
        if len(self.level_downsamples) <= 1:
            return None
        cands = [(i, float(d[0])) for i, d in enumerate(self.level_downsamples) if i >= 1]
        # tie-break: prefer larger downsample (coarser)
        best_i = min(cands, key=lambda t: (abs(t[1] - target_downsample), -t[1]))[0]
        return best_i

    # ===================== Harmonization & Seg Planning ======================

    def set_harmonization(self, harmonize=True, target_mpp=0.25, desired_seg_downsample=32.0):
        self.harmonize = bool(harmonize)
        self.target_mpp = float(target_mpp)
        self.harmonization_skipped = False  # NEW FLAG

        # === STRICT REQUIREMENT: skip if MPP is missing ===
        if self.harmonize and (not self.mpp_x or not self.mpp_y):
            self.dx_target = None
            self.dy_target = None
            self.seg_level = 0
            self.seg_down = 1.0
            self.seg_is_virtual = False
            self.dx_seg = None
            self.dy_seg = None
            self.harmonization_skipped = True
            print("[HARMONIZATION] ERROR: Missing MPP metadata (mpp_x or mpp_y). Skipping this slide.")
            return  # do not plan segmentation / visualization

        # (unchanged when MPP exists)
        self.dx_target = self.target_mpp / self.mpp_x
        self.dy_target = self.target_mpp / self.mpp_y

        nz = self._closest_nonzero_level(desired_seg_downsample)
        if nz is not None:
            self.seg_level = int(nz)
            self.seg_down = float(self.level_downsamples[nz][0])
            self.seg_is_virtual = False
            print(f"[SEG] Using native nonzero level: {self.seg_level} (down≈{self.seg_down:.3f})")
        else:
            self.seg_level = 0
            self.seg_down = float(max(1.0, desired_seg_downsample))
            self.seg_is_virtual = True
            print(f"[SEG] Single-level slide: Synthesize digital segmentation level down={self.seg_down:.3f}")

        self.dx_seg = self.dx_target / self.seg_down
        self.dy_seg = self.dy_target / self.seg_down

        print("\n[HARMONIZATION] =======================")
        print(f"[HARMONIZATION] harmonize     : {self.harmonize}")
        print(f"[HARMONIZATION] target_mpp    : {self.target_mpp:.6f}")
        print(f"[HARMONIZATION] dx_target/dy  : {self.dx_target:.6f} / {self.dy_target:.6f}")
        print(f"[HARMONIZATION] seg_level     : {self.seg_level}  (virtual={self.seg_is_virtual})")
        print(f"[HARMONIZATION] seg_down      : {self.seg_down:.6f}")
        print(f"[HARMONIZATION] dx_seg/dy_seg : {self.dx_seg:.6f} / {self.dy_seg:.6f}")
        print("[HARMONIZATION] =======================\n")

    def harmonized_patch_from(self, patch_size, force_square=False, square_mode="max"):
        """
        Given a target-space patch_size (at target_mpp), return native anisotropic size (pw, ph, sq_suggest).
        """
        from math import sqrt
        assert self.dx_target is not None and self.dy_target is not None, "Call set_harmonization() first."
        pw = max(1, int(round(patch_size * float(self.dx_target))))
        ph = max(1, int(round(patch_size * float(self.dy_target))))
        if not force_square:
            sq = max(pw, ph)
            print(f"[HARMONIZATION] harmonized_patch (anisotropic): {pw}x{ph}  (sq_suggest={sq})")
            return pw, ph, sq
        if square_mode == "min":
            sq = min(pw, ph)
        elif square_mode == "mean":
            sq = int(round((pw + ph) * 0.5))
        elif square_mode == "geom":
            sq = int(round(sqrt(pw * ph)))
        else:
            sq = max(pw, ph)
        print(f"[HARMONIZATION] harmonized_patch (forced square={square_mode}): {sq}")
        return pw, ph, max(1, sq)

    # ============================ Original API ===========================

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array(
                [[[int(float(coord.attributes['X'].value)),
                   int(float(coord.attributes['Y'].value))]] for coord in coord_list],
                dtype='int32'
            )
        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        self.contours_tumor = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self, annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)
                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1,1,2)
                        all_cnts.append(contour)
            return all_cnts

        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        self.contours_tumor = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    # ----------------------------- SEGMENTATION ------------------------------

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
                      filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8}, ref_patch_size=512,
                      exclude_ids=[], keep_ids=[]):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold.
        Never runs at L0; uses virtual proxy for single-level slides.
        """
        def _filter_contours(contours, hierarchy, filter_params):
            filtered = []
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if a > float(filter_params['a_t']):
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = [hole for hole in unfilered_holes if cv2.contourArea(hole) > float(filter_params['a_h'])]
                hole_contours.append(filtered_holes)
            return foreground_contours, hole_contours

        W0, H0 = self.level_dim[0]
        print(f"[SEG] performing SEGMENTATION; requested seg_level={seg_level}, virtual={self.seg_is_virtual}")

        # enforce "never L0"
        if self.seg_is_virtual:
            Wv = max(1, int(math.floor(W0 / self.seg_down)))
            Hv = max(1, int(math.floor(H0 / self.seg_down)))
            print(f"[SEG] Virtual pyramid: down={self.seg_down:.3f} → proxy size {Wv}×{Hv}")
            img_pil = self.backend.get_downscaled_full((Wv, Hv), prefer_rgba=True)
            seg_scale = (self.seg_down, self.seg_down)
        else:
            assert self.seg_level >= 1, "Internal: seg_level must be >=1 when using a native level."
            print(f"[SEG] Native seg level {self.seg_level} dims {self.level_dim[self.seg_level]}")
            img_pil = self.read_region((0, 0), self.seg_level, self.level_dim[self.seg_level])
            seg_scale = self.level_downsamples[self.seg_level]

        print(f"[SEG] Image mode: {img_pil.mode}")
        img = np.array(img_pil)
        rgb = img[:, :, :3] if (img.ndim == 3 and img.shape[2] == 4) else img
        print(f"[SEG] Input shape: {img.shape}, dtype: {img.dtype}, min={img.min()}, max={img.max()}")

        img_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)

        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        # scale back to L0 geometry
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (seg_scale[0] * seg_scale[1]))
        fparams = dict(filter_params)
        fparams['a_t'] = fparams['a_t'] * scaled_ref_patch_area
        fparams['a_h'] = fparams['a_h'] * scaled_ref_patch_area
        print(f"[SEG] ref_patch_size={ref_patch_size}")
        print(f"[SEG] scale used={seg_scale}")
        print(f"[SEG] scaled_ref_patch_area={scaled_ref_patch_area}")

        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if fparams:
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, fparams)

        # Map contours from proxy coords → L0 by multiplying by seg_scale
        self.contours_tissue = self.scaleContourDim(foreground_contours, seg_scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, seg_scale)

        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]
        print(f"[SEG] tissue contours kept: {len(self.contours_tissue)}")

    # ----------------------------- HARMONIZATION ASSETS ----------------------
    def _pick_tissue_center_l0(self, desired_tile_w=896, desired_tile_h=896, thumb_max=2048):
        """
        Super-simple & robust tissue center picker.

        Strategy:
        - Sample a few small patches (<=512x512) at a 3x3 grid of positions.
        - Use isBlackPatch / isWhitePatch to reject obvious background.
        - Return the center of the first "non-background" patch.
        - If nothing passes, fall back to slide center.

        Returns (cx, cy) in L0 coordinates.
        """
        import numpy as np
        import time

        W0, H0 = self.level_dim[0]

        # Size of small test patch (never huge, no thumbnails)
        sample_size = int(min(512, desired_tile_w, desired_tile_h, W0, H0))
        if sample_size <= 0:
            # completely degenerate case, just return slide center
            return W0 // 2, H0 // 2

        # Helper: clamp a chosen center so that a tile of desired_tile_* fits
        def _clamp_center(cx, cy):
            # X
            if W0 <= desired_tile_w:
                cx_out = W0 // 2
            else:
                cx_min = desired_tile_w // 2
                cx_max = W0 - desired_tile_w // 2
                cx_out = int(np.clip(cx, cx_min, cx_max))
            # Y
            if H0 <= desired_tile_h:
                cy_out = H0 // 2
            else:
                cy_min = desired_tile_h // 2
                cy_max = H0 - desired_tile_h // 2
                cy_out = int(np.clip(cy, cy_min, cy_max))
            return int(cx_out), int(cy_out)

        # 3x3 grid of fractional positions (center, edges, corners)
        frac_positions = [
            (0.5, 0.5),
            (0.25, 0.5), (0.75, 0.5),
            (0.5, 0.25), (0.5, 0.75),
            (0.25, 0.25), (0.75, 0.25),
            (0.25, 0.75), (0.75, 0.75),
        ]

        print(f"[QA] Tissue center picker: sampling {len(frac_positions)} small patches of {sample_size}x{sample_size}", flush=True)

        for i, (fx, fy) in enumerate(frac_positions):
            cx = int(W0 * fx)
            cy = int(H0 * fy)

            # top-left for the small sampling patch, clamped inside the slide
            x0 = int(np.clip(cx - sample_size // 2, 0, max(0, W0 - sample_size)))
            y0 = int(np.clip(cy - sample_size // 2, 0, max(0, H0 - sample_size)))

            t0 = time.time()
            try:
                patch_pil = self.read_region((x0, y0), 0, (sample_size, sample_size)).convert("RGB")
            except Exception as e:
                print(f"[QA] Sample {i}: read_region failed at ({x0},{y0}) with {e}, skipping", flush=True)
                continue
            t1 = time.time()

            arr = np.array(patch_pil)

            # Use existing helpers to discard obvious background
            if isBlackPatch(arr, rgbThresh=50) or isWhitePatch(arr, satThresh=15):
                print(f"[QA] Sample {i}: ({fx:.2f},{fy:.2f}) looks background (time {t1 - t0:.3f}s), skipping", flush=True)
                continue

            # Looks like tissue → accept this patch center
            cx_real = x0 + sample_size // 2
            cy_real = y0 + sample_size // 2
            cx_clamped, cy_clamped = _clamp_center(cx_real, cy_real)
            print(f"[QA] Sample {i}: ({fx:.2f},{fy:.2f}) accepted as tissue center at ({cx_clamped},{cy_clamped}) "
                  f"(read {t1 - t0:.3f}s)", flush=True)
            return cx_clamped, cy_clamped

        # If all samples failed → fallback to slide center
        print("[QA] No clear tissue found in sampled patches; falling back to slide center", flush=True)
        cx_center = W0 // 2
        cy_center = H0 // 2
        return _clamp_center(cx_center, cy_center)


    def save_harmonization_assets(self, base_save_dir, visualization_patch_size=448):
        """
        Writes to <base_save_dir>/harmonization/<slide>/ :
        - <slide>_report.json
        - <slide>_before_native.png
        - <slide>_after_native.png
        - <slide>_viz.jpg  (side-by-side with 10 µm grid, per-panel scale bars, titles, and info)
        """
        import time
        start_total = time.time()
        print(f"[QA] === Starting harmonization assets for {self.name} ===", flush=True)

        out_dir = os.path.join(base_save_dir, "harmonization", self.name)
        os.makedirs(out_dir, exist_ok=True)

        # MPP must exist
        if not (self.mpp_x and self.mpp_y):
            print("[HARMONIZATION] SKIP: MPP metadata missing; not generating QA or report.")
            with open(os.path.join(out_dir, f"{self.name}_SKIPPED_missing_mpp.txt"), "w") as f:
                f.write("Skipped harmonization QA/report: missing mpp_x/mpp_y metadata.\n")
            return False

        dx = float(self.dx_target if self.dx_target else self.target_mpp / self.mpp_x)
        dy = float(self.dy_target if self.dy_target else self.target_mpp / self.mpp_y)

        report = {
            "slide": self.name,
            "kind": getattr(self.backend, "kind", None),
            "dim_L0": self.level_dim[0],
            "levels": len(self.level_dim),
            "downs": self.level_downsamples,
            "mpp_x": self.mpp_x, "mpp_y": self.mpp_y, "objective": self.objective,
            "harmonize": self.harmonize, "target_mpp": self.target_mpp,
            "dx_target": dx, "dy_target": dy,
            "seg_level": self.seg_level, "seg_is_virtual": self.seg_is_virtual,
            "seg_down": self.seg_down, "dx_seg": getattr(self, "dx_seg", None), "dy_seg": getattr(self, "dy_seg", None)
        }

        # --- Choose a tissue center (unchanged) ---
        px_h = max(1, int(round(visualization_patch_size * dx)))
        py_h = max(1, int(round(visualization_patch_size * dy)))
        half_w = max(visualization_patch_size // 2, px_h // 2)
        half_h = max(visualization_patch_size // 2, py_h // 2)

        print(f"[QA] Picking tissue center...", flush=True)
        t0 = time.time()
        cx, cy = self._pick_tissue_center_l0(desired_tile_w=2 * half_w, desired_tile_h=2 * half_h)
        t1 = time.time()
        print(f"[QA] Tissue center at ({cx}, {cy}) | took {t1 - t0:.2f}s", flush=True)

        # --- BEFORE patch (unchanged) ---
        x0_b = cx - visualization_patch_size // 2
        y0_b = cy - visualization_patch_size // 2
        print("[QA] Reading BEFORE patch...", flush=True)
        before_native = self.read_region((x0_b, y0_b), 0, (visualization_patch_size, visualization_patch_size)).convert("RGB")
        t2 = time.time()
        print(f"[QA] BEFORE read done in {t2 - t1:.2f}s", flush=True)

        # --- AFTER patch (unchanged) ---
        x0_a = cx - px_h // 2
        y0_a = cy - py_h // 2
        print("[QA] Reading AFTER patch...", flush=True)
        after_native = self.read_region((x0_a, y0_a), 0, (px_h, py_h)).convert("RGB")
        t3 = time.time()
        print(f"[QA] AFTER read done in {t3 - t2:.2f}s", flush=True)

        print("[QA] Resizing AFTER patch (BILINEAR)...", flush=True)
        after_np_resized = np.array(after_native.resize((visualization_patch_size, visualization_patch_size), Image.BILINEAR))
        t4 = time.time()
        print(f"[QA] Resize done in {t4 - t3:.2f}s", flush=True)

        # --- Save raw crops (unchanged) ---
        before_native.save(os.path.join(out_dir, f"{self.name}_before_native.png"), quality=85)
        after_native.save(os.path.join(out_dir,  f"{self.name}_after_native.png"),  quality=85)

        # --- Grid overlays (10 µm) + SSIM ---
        before_np = np.array(before_native)
        um_per_px_before = float((self.mpp_x + self.mpp_y) * 0.5)
        um_per_px_after  = float(self.target_mpp)
        GRID_UM = 10.0

        before_grid = _grid_overlay_um(before_np,        um_per_px=um_per_px_before, um_step=GRID_UM, fallback_px=28)
        after_grid  = _grid_overlay_um(after_np_resized, um_per_px=um_per_px_after,  um_step=GRID_UM, fallback_px=28)

        sim_val, sim_kind = _compute_ssim_01(before_np, after_np_resized)
        report["similarity_metric"] = sim_kind
        report["ssim_before_after"] = sim_val
        t5 = time.time()
        print(f"[QA] Grids + SSIM computed in {t5 - t4:.2f}s (SSIM={sim_val:.4f})", flush=True)

        with open(os.path.join(out_dir, f"{self.name}_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # --- If viz disabled, stop here (respect existing flag) ---
        if not getattr(self, "qa_viz", True):
            print(f"[QA] Visualization skipped (qa_viz=False). Total {time.time() - start_total:.2f}s", flush=True)
            return True

        # --- Compose visualization: side-by-side + scale bars + titles + info ---
        pad = 24
        line_h = 20
        img_w = visualization_patch_size * 2 + pad * 3
        top_h = pad + visualization_patch_size + pad

        font       = _load_font(18)
        font_small = _load_font(16)

        # Top panel with grid overlays
        canvas = Image.new("RGB", (img_w, top_h), (20, 20, 20))
        canvas.paste(Image.fromarray(before_grid), (pad, pad))
        canvas.paste(Image.fromarray(after_grid),  (pad * 2 + visualization_patch_size, pad))
        draw = ImageDraw.Draw(canvas)

        # Per-panel scale bars  —— force a fixed 10 µm reference on both panels
        SCALE_UM = 10.0
        for umpp, xoff in ((um_per_px_before, pad), (um_per_px_after, pad * 2 + visualization_patch_size)):
            # convert fixed physical length to pixels for this panel
            px_len = max(1, int(round(SCALE_UM / float(umpp))))
            # position near bottom-right, with same margins as before
            x1 = xoff + visualization_patch_size - 12
            x0 = x1 - px_len
            y1 = pad + visualization_patch_size - 12
            y0 = y1 - 8
            draw.rectangle([x0, y0, x1, y1], fill=(240, 240, 240))

            label = f"{int(SCALE_UM) if SCALE_UM >= 10 else SCALE_UM:g} µm"
            tw = draw.textlength(label, font=font_small)
            draw.text((x0 + (px_len - tw) / 2.0, y0 - 18), label, fill=(230, 230, 230), font=font_small)

            # Titles + 10 µm legend (wrapped per panel to avoid overlap)
            left_title  = "Before: native 448×448"
            right_title = f"After: harmonized to {self.target_mpp:.3f} µm/px (reads {px_h}×{py_h} native → 448)"
            legend      = "Grid spacing: 10 µm"

            max_title_w = visualization_patch_size

            # build once to measure/wrap
            tmp_img = Image.new("RGB", (10, 10))
            tmp_draw = ImageDraw.Draw(tmp_img)

            lt_lines = _wrap(tmp_draw, left_title,  font, max_title_w)
            rt_lines = _wrap(tmp_draw, right_title, font, max_title_w)
            n_lines  = max(len(lt_lines), len(rt_lines))

            titles_h = n_lines * line_h + line_h + pad  # +1 line for legend
            titles_img = Image.new("RGB", (img_w, titles_h), (20, 20, 20))
            d2 = ImageDraw.Draw(titles_img)

            # left column
            yL = 0
            for t in lt_lines:
                d2.text((pad, yL), t, fill=(235, 235, 235), font=font)
                yL += line_h

            # right column
            xR = pad * 2 + visualization_patch_size
            yR = 0
            for t in rt_lines:
                d2.text((xR, yR), t, fill=(235, 235, 235), font=font)
                yR += line_h

            # centered legend on its own line beneath titles
            tw = d2.textlength(legend, font=font_small)
            d2.text(((img_w - tw) / 2, n_lines * line_h + 4), legend, fill=(215, 215, 215), font=font_small)

        # Compact info block
        info_lines = [
            f"Similarity ({sim_kind.upper()}): {sim_val:.4f}  (1.0 = identical)",
            f"mpp_x={self.mpp_x:.4f}  mpp_y={self.mpp_y:.4f}  target_mpp={self.target_mpp:.4f}",
            f"seg_level={self.seg_level} (virtual={self.seg_is_virtual})  seg_down={self.seg_down:.2f}",
            f"dx_target={dx:.4f}  dy_target={dy:.4f}",
        ]
        info_h = pad + line_h * len(info_lines)
        info_img = Image.new("RGB", (img_w, info_h), (20, 20, 20))
        d3 = ImageDraw.Draw(info_img)
        y = pad // 2
        for L in info_lines:
            d3.text((pad, y), L, fill=(235,235,235), font=font_small)
            y += line_h

        # Stack & save
        out = Image.new("RGB", (img_w, top_h + titles_h + info_h), (20, 20, 20))
        out.paste(canvas,     (0, 0))
        out.paste(titles_img, (0, top_h))
        out.paste(info_img,   (0, top_h + titles_h))

        out_jpg = os.path.join(out_dir, f"{self.name}_viz.jpg")
        out.save(out_jpg, quality=92)
        t6 = time.time()
        print(f"[viz] harmonization assets written → {out_dir}")
        print(
            f"[QA TIMING] pick_center={t1-t0:.3f}s, "
            f"read_before={t2-t1:.3f}s ({visualization_patch_size}x{visualization_patch_size}), "
            f"read_after={t3-t2:.3f}s ({px_h}x{py_h}), "
            f"resize={t4-t3:.3f}s, grids+ssim={t5-t4:.3f}s, "
            f"write_jpg={t6-t5:.3f}s, total={t6-t0:.3f}s"
        )
        return True

    # Backward-compat wrapper (keeps existing driver calls working)
    def save_harmonization_viz(self, out_path, visualization_patch_size=448, compare_mode="target"):
        base_dir = os.path.dirname(os.path.dirname(out_path)) if out_path else "."
        self.save_harmonization_assets(base_dir, visualization_patch_size=visualization_patch_size)

    # ----------------------------- VISUALIZATION -----------------------------

    def visWSI(self, vis_level=0, color=(0,255,0), hole_color=(0,0,255), annot_color=(255,0,0),
               line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
               view_slide_only=False, number_contours=False, seg_display=True, annot_display=True):
        """
        Preview image with overlays.
        Never uses L0 directly: pick a real level >=1 when available; else proxy.
        """

        # --- NEW GUARD ---
        if vis_level is not None and vis_level >= self.level_count:
            raise ValueError(f"vis_level {vis_level} out of range for slide with {self.level_count} levels")

        # Choose rendering level (never 0)
        if vis_level <= 0:
            nz = self._closest_nonzero_level(32.0)
            vis_level = nz if nz is not None else None  # None => proxy path

        # Compute region size and scale
        if vis_level is not None:
            downsample = self.level_downsamples[vis_level]
            scale = [1/downsample[0], 1/downsample[1]]
            if top_left is not None and bot_right is not None:
                top_left = tuple(top_left)
                bot_right = tuple(bot_right)
                w, h = tuple((np.array(bot_right) * np.array(scale)).astype(int) -
                             (np.array(top_left) * np.array(scale)).astype(int))
                region_size = (max(1, w), max(1, h))
            else:
                top_left = (0,0)
                region_size = self.level_dim[vis_level]
            img = np.array(self.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # proxy from L0 (single-level slides)
            W0, H0 = self.level_dim[0]
            proxy_down = 32.0
            Wp = max(1, int(round(W0 / proxy_down)))
            Hp = max(1, int(round(H0 / proxy_down)))
            region_size = (Wp, Hp)
            scale = [1/proxy_down, 1/proxy_down]
            top_left = (0,0)
            img = np.array(self.backend.get_downscaled_full(region_size, prefer_rgba=False).convert("RGB"))

        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * np.array(scale)).astype(int))
            lt = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, lt, lineType=cv2.LINE_8, offset=offset)
                else:
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        cv2.drawContours(img, [contour], -1, color, lt, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale),
                                 -1, annot_color, lt, lineType=cv2.LINE_8, offset=offset)

        img = Image.fromarray(img)
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
        return img

    # ------------------------------- VIEW & PATCH ----------------------------

    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        print("Creating patches for: ", self.name, "...")
        _t0 = time.time()

        # --- NEW GUARD ---
        if patch_level < 0 or patch_level >= self.level_count:
            raise ValueError(f"patch_level {patch_level} out of range for slide with {self.level_count} levels")

        # Clamp step_size
        if step_size > patch_size:
            print(f"[PATCH] step_size ({step_size}) > patch_size ({patch_size}) — clamping to patch_size.")
            step_size = patch_size

        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)
                except StopIteration:
                    continue
                self.hdf5_file = initialize_hdf5_bag(first_patch, save_coord=save_coord)

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file

    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, custom_downsample=1,
                           white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if step_size > patch_size:
            print(f"[PATCH] step_size ({step_size}) > patch_size ({patch_size}) — clamping to patch_size.")
            step_size = patch_size

        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print(f"Custom Downsample: {custom_downsample}, Patching at {patch_size}x{patch_size}, Final {target_patch_size}x{target_patch_size}")

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1])
            stop_x = min(start_x + w, img_w - ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, (x, y), self.holes_tissue[cont_idx], ref_patch_size[0]):
                    continue

                count += 1
                patch_PIL = self.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                if white_black:
                    arr = np.array(patch_PIL)
                    if isBlackPatch(arr, rgbThresh=black_thresh) or isWhitePatch(arr, satThresh=white_thresh):
                        continue

                patch_info = {
                    'x': x // (patch_downsample[0] * custom_downsample),
                    'y': y // (patch_downsample[1] * custom_downsample),
                    'cont_idx': cont_idx, 'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level]) // custom_downsample),
                    'level_dim': self.level_dim[patch_level],
                    'patch_PIL': patch_PIL, 'name': self.name, 'save_path': save_path
                }

                yield patch_info

        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        """
        Return 1 if the patch center (pt + patch_size/2) lies inside ANY hole contour, else 0.
        """
        if not holes:
            return 0
        cx = pt[0] + (patch_size / 2.0)
        cy = pt[1] + (patch_size / 2.0)
        for hole in holes:
            if hole.dtype != np.int32:
                hole = hole.astype(np.int32, copy=False)
            if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
                return 1
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        downs = []
        dim_0 = self.backend.level_dimensions[0]
        for i, dim in enumerate(self.backend.level_dimensions):
            base_w, base_h = dim_0
            w, h = dim
            if i < len(self.backend.level_downsamples):
                ds = self.backend.level_downsamples[i]
                if isinstance(ds, (tuple, list)) and len(ds) == 2:
                    downs.append((float(ds[0]), float(ds[1])))
                else:
                    downs.append((float(ds), float(ds)))
            else:
                estimated_downsample = (base_w / float(w), base_h / float(h))
                downs.append(estimated_downsample)
        return downs

    # ------------------------------- PROCESSING ------------------------------

    def process_contours(self, save_path, patch_level, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...")
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = max(1, math.ceil(n_contours * 0.05))
        init = True

        # --- NEW GUARD ---
        if patch_level < 0 or patch_level >= self.level_count:
            raise ValueError(f"patch_level {patch_level} out of range for slide with {self.level_count} levels")

        if step_size > patch_size:
            print(f"[PATCH] step_size ({step_size}) > patch_size ({patch_size}) — clamping to patch_size.")
            step_size = patch_size

        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))

            asset_dict, attr_dict = self.process_contour(
                cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs
            )
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file

    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size=256, step_size=256,
                        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        if step_size > patch_size:
            print(f"[PATCH] step_size ({step_size}) > patch_size ({patch_size}) — clamping to patch_size.")
            step_size = patch_size

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        print(f"self.level_downsamples[patch_level] {self.level_downsamples[patch_level]}")
        print(f"patch_level {patch_level}")
        print(f"patch_downsample {patch_downsample}")
        print(f"ref_patch_size {ref_patch_size}")
        print(f"The actual analysis uses ref_patch_size; tiles are resized to {patch_size} on save.")

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = min(4, mp.cpu_count())
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        if len(results) > 0:
            asset_dict = {'coords': results}
            attr = {'patch_size': patch_size,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                    'level_dim': self.level_dim[patch_level],
                    'name': self.name,
                    'save_path': save_path}
            attr_dict = {'coords': attr}
            return asset_dict, attr_dict
        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    # ------------------------------- HEATMAP ---------------------------------

    def visHeatmap(self, scores, coords, vis_level=-1,
                   top_left=None, bot_right=None,
                   patch_size=(256, 256),
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
                   blur=False, overlap=0.0,
                   segment=True, use_holes=True,
                   convert_to_percentiles=False,
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample=1,
                   cmap='coolwarm'):

        # choose nonzero level by default; proxy if single-level
        if vis_level < 0:
            nz = self._closest_nonzero_level(32.0)
            vis_level = nz if nz is not None else 0

        # --- NEW GUARD ---
        if vis_level >= self.level_count:
            raise ValueError(f"vis_level {vis_level} out of range for slide with {self.level_count} levels")

        downsample = self.level_downsamples[vis_level] if vis_level is not None else (1.0, 1.0)
        scale = [1 / downsample[0], 1 / downsample[1]]

        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            threshold = 1.0 / len(scores) if thresh < 0 else thresh
        else:
            threshold = 0.0

        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * np.array(scale)).astype(int) -
                         (np.array(top_left) * np.array(scale)).astype(int))
            region_size = (w, h)
        else:
            region_size = self.level_dim[vis_level] if vis_level is not None else self.level_dim[0]
            top_left = (0, 0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        if convert_to_percentiles:
            scores = to_percentiles(scores)
        scores /= 100

        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score = 1.0
                    count += 1
            else:
                score = 0.0
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))

        zero_mask = counter == 0
        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter
        if blur:
            overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))

        if not blank_canvas:
            img = np.array(self.read_region(top_left, vis_level if vis_level is not None else 0, region_size).convert("RGB"))
        else:
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))

            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
                color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)
                if segment:
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                    img_block[mask_block] = color_block[mask_block]
                else:
                    img_block = color_block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()

        print('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if alpha < 1.0:
            img = self.block_blending(img, vis_level if vis_level is not None else 0, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)

        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)
                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue

                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)
                if not blank_canvas:
                    pt = (x_start, y_start)
                    canvas = np.array(self.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0, 0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)
            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)

        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask
