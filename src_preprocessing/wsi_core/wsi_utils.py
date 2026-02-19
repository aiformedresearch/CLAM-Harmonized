# wsi_core/wsi_utils.py
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import math
import cv2
from PIL import Image
from tqdm import tqdm

# =============================== Basic Patch Tests ===============================

def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

# =============================== Small Helpers ===================================

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt': (x, y)})
                yield process_dict
            else:
                yield (x, y)

def _safe_level_dimensions(wsi_like):
    """Return list of (W,H) for levels."""
    if hasattr(wsi_like, 'level_dimensions'):
        return list(wsi_like.level_dimensions)
    if hasattr(wsi_like, 'level_dim'):
        return list(wsi_like.level_dim)
    raise AttributeError("WSI object lacks level dimensions.")

def _safe_level_downsamples(wsi_like):
    """Return list/tuple of per-level downsample factors (dx,dy) tuples."""
    if hasattr(wsi_like, 'level_downsamples'):
        return list(wsi_like.level_downsamples)
    # fallback: estimate from geometry
    L = _safe_level_dimensions(wsi_like)
    w0, h0 = L[0]
    outs = []
    for (w, h) in L:
        dx = w0 / float(w)
        dy = h0 / float(h)
        outs.append((dx, dy))
    return outs

def _safe_get_best_level_for_downsample(wsi_like, target_down=16.0):
    """
    Try wsi_like.get_best_level_for_downsample if present, else pick level with
    downsample closest to target_down.
    """
    if hasattr(wsi_like, 'get_best_level_for_downsample'):
        try:
            return int(wsi_like.get_best_level_for_downsample(target_down))
        except Exception:
            pass
    downs = _safe_level_downsamples(wsi_like)
    ds = [d[0] if isinstance(d, (list, tuple)) else float(d) for d in downs]
    return int(np.argmin([abs(d - float(target_down)) for d in ds]))

# =============================== HDF5 Save Utils =================================

def savePatchIter_bag_hdf5(patch):
    """
    Append one patch to the HDF5 bag. The 'patch' is a dict produced by the patch generator:
      {
        'x': ..., 'y': ..., 'cont_idx': ..., 'patch_level': ...,
        'downsample': (dx,dy),
        'downsampled_level_dim': (W_l,H_l),
        'level_dim': (W_l,H_l),
        'patch_PIL': PIL.Image,
        'name': slide_name, 'save_path': dir
      }
    """
    # Robust unpack by keys (avoid relying on dict insertion order)
    x = patch['x']; y = patch['y']
    cont_idx = patch['cont_idx']; patch_level = patch['patch_level']
    downsample = patch['downsample']; downsampled_level_dim = patch['downsampled_level_dim']
    level_dim = patch['level_dim']; img_patch = patch['patch_PIL']
    name = patch['name']; save_path = patch['save_path']

    img_patch = np.array(img_patch)[np.newaxis, ...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, "a")

    if 'imgs' not in file:
        # create dataset if missing (rare, but safe)
        dset = file.create_dataset('imgs',
                                   shape=img_patch.shape,
                                   maxshape=(None,) + img_patch.shape[1:],
                                   chunks=img_patch.shape,
                                   dtype=img_patch.dtype)
        dset[:] = img_patch
        # add coords dataset if present in first call
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2),
                                         chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x, y)
    else:
        dset = file['imgs']
        dset.resize(len(dset) + img_shape[0], axis=0)
        dset[-img_shape[0]:] = img_patch

        if 'coords' in file:
            coord_dset = file['coords']
            coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
            coord_dset[-img_shape[0]:] = (x, y)

    file.close()

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None and key in attr_dict:
                for attr_key, attr_val in attr_dict[key].items():
                    dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    x = first_patch['x']; y = first_patch['y']
    patch_level = first_patch['patch_level']
    downsample = first_patch['downsample']
    level_dim = first_patch['level_dim']
    downsampled_level_dim = first_patch['downsampled_level_dim']
    img_patch = np.array(first_patch['patch_PIL'])[np.newaxis, ...]
    name = first_patch['name']; save_path = first_patch['save_path']

    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, "w")

    dtype = img_patch.dtype
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:]
    dset = file.create_dataset('imgs', shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=dtype)
    dset[:] = img_patch

    print(f"[HDF5:init] x={x}, y={y}, patch_level={patch_level}, downsample={downsample}, "
          f"level_dim={level_dim}, downsampled_level_dim={downsampled_level_dim}")
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x, y)

    file.close()
    return file_path

# =============================== Sampling & Scoring ==============================

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids = scores.argsort()[:k]
    else:
        top_k_ids = scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):
    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

# =============================== Stitching Helpers ===============================

def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        (0, 0, 0, 255),
        thickness=thickness,
    )
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = max(1, math.ceil(total * 0.1))
        print('[STITCH] start stitching {}'.format(patch_dset.attrs.get('wsi_name', '<unknown>')))

    for idx in range(total):
        if verbose > 0 and idx % ten_percent_chunk == 0:
            print('[STITCH] progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size_native, vis_level, indices=None, draw_grid=True):
    """
    Args:
      canvas: np.array HxWx3/4 (pre-allocated)
      wsi_object: WholeSlideImage (proxy) or OpenSlide-like; must have read_region
      coords: (N,2) integer coordinates in level-0 space
      patch_size_native: (pw,ph) in level-0 (reference) pixels
      vis_level: level to read (downsampled view); if -1 -> use virtual view via wsi_object.vis_down
    """
    import numpy as np
    from PIL import Image

    wsi = wsi_object.getOpenSlide() if hasattr(wsi_object, 'getOpenSlide') else wsi_object
    level_dims = _safe_level_dimensions(wsi)
    downsamples = _safe_level_downsamples(wsi)

    # Determine how we map L0 -> visualization canvas
    virtual_view = (vis_level == -1) or getattr(wsi_object, "vis_is_virtual", False)
    if virtual_view:
        # Use the same downsample chosen for segmentation (or a provided vis_down)
        vis_down = float(getattr(wsi_object, "vis_down", getattr(wsi_object, "seg_down", 32.0)))
        dsx, dsy = vis_down, vis_down
        print(f"[STITCH] virtual view: down={vis_down:.2f}", flush=True)
    else:
        dsx, dsy = downsamples[vis_level]
        # dsx/dsy may be tuples/floats already; coerce to floats
        dsx, dsy = float(dsx), float(dsy)
        print(f"[STITCH] native view: level={vis_level}  ds≈({dsx:.2f},{dsy:.2f})", flush=True)

    # Convert native patch size (L0) to vis-level pixel size
    patch_size = tuple(np.ceil((np.array(patch_size_native, dtype=np.float64) / np.array([dsx, dsy], dtype=np.float64))).astype(np.int32))
    print('[STITCH] downscaled patch size for stitching visualization: {}x{}'.format(patch_size[0], patch_size[1]))
    print('[STITCH] total coords: ', len(coords))
    try:
        base_wh = level_dims[0]
        print('[STITCH] L0 dimensions: {} x {}'.format(base_wh[0], base_wh[1]))
    except Exception:
        pass

    if indices is None:
        indices = np.arange(len(coords), dtype=np.int64)
    total = len(indices)

    Hc, Wc = canvas.shape[:2]

    # Main loop
    for idx in tqdm(range(total)):
        patch_id = int(indices[idx])
        x0, y0 = map(int, coords[patch_id])  # L0 coords

        # Place using vis-level coordinates (integer)
        x_vis = int(np.floor(x0 / dsx))
        y_vis = int(np.floor(y0 / dsy))

        # Read a native patch (L0). For virtual: read L0 then downscale to patch_size.
        # For native: ask backend at vis_level directly sized to patch_size.
        try:
            if virtual_view:
                # Read at L0 with native size, then resize to vis patch_size
                pw_native, ph_native = map(int, patch_size_native)
                # Clip read window at slide bounds to avoid backend errors
                # (PIL/OpenSlide handle partial reads, but we clip for safety)
                x_read, y_read = x0, y0
                if base_wh[0] and base_wh[1]:
                    x_read = max(0, min(x_read, base_wh[0] - 1))
                    y_read = max(0, min(y_read, base_wh[1] - 1))
                patch_img = wsi.read_region((x_read, y_read), 0, (pw_native, ph_native)).convert("RGB")
                # Resize to vis patch size with BILINEAR (fast, good enough for viz)
                patch_img = patch_img.resize((int(patch_size[0]), int(patch_size[1])), Image.BILINEAR)
            else:
                # Native pyramid: ask directly at vis_level with vis patch size
                patch_img = wsi.read_region((x0, y0), int(vis_level), (int(patch_size[0]), int(patch_size[1]))).convert("RGB")

            patch = np.array(patch_img, copy=False)
        except Exception as e:
            # If a patch fails (edge cases), skip gracefully
            if idx == 0 or (idx % 1000 == 0):
                print(f"[STITCH][WARN] read_region failed at idx={idx} ({x0},{y0}) -> {e}", flush=True)
            continue

        # Compute destination region in canvas with clipping
        x1 = x_vis
        y1 = y_vis
        x2 = x_vis + patch.shape[1]
        y2 = y_vis + patch.shape[0]

        # Clip against canvas bounds
        if x2 <= 0 or y2 <= 0 or x1 >= Wc or y1 >= Hc:
            continue
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = min(Wc, x2); cy2 = min(Hc, y2)

        # Corresponding crop in the patch
        px1 = cx1 - x1; py1 = cy1 - y1
        px2 = px1 + (cx2 - cx1); py2 = py1 + (cy2 - cy1)

        # Blit onto canvas
        canvas[cy1:cy2, cx1:cx2, :3] = patch[py1:py2, px1:px2, :]

        # Optional grid
        if draw_grid:
            DrawGrid(canvas, (cx1, cy1), (cx2 - cx1, cy2 - cy1))

    return Image.fromarray(canvas)

# =============================== Top-Level Stitchers =============================

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['imgs']
        coords = file['coords'][:]
        if 'downsampled_level_dim' in dset.attrs.keys():
            w, h = dset.attrs['downsampled_level_dim']
        else:
            w, h = dset.attrs['level_dim']

    print('[STITCH] original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h // downscale
    coords = (coords / downscale).astype(np.int32)
    print('[STITCH] downscaled size for stiching: {} x {}'.format(w, h))
    print(f'[STITCH] number of patches: {len(coords)}')
    img_shape = dset[0].shape
    print('[STITCH] patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)
    print(f'[STITCH] downscaled patch shape: {downscaled_shape} with downscale {downscale}')

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(f"Visualization Downscale %d is too large: w={w}, h={h}, w*h={w*h}, limit: {Image.MAX_IMAGE_PIXELS}" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)

    return heatmap

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    """
    Stitch by re-reading from the slide (more accurate for visualization).
    Visualization downsample is tied to the segmentation plan:
    - if seg_is_virtual, use seg_down as the base
    - else use nearest native level to requested `downscale`
    Auto-shrinks further if the canvas would exceed PIL's pixel limit.
    """
    import math
    import h5py
    import numpy as np
    from PIL import Image

    # Prefer a backend object if exposed by the wrapper
    wsi = wsi_object.getOpenSlide() if hasattr(wsi_object, 'getOpenSlide') else wsi_object
    level_dims = _safe_level_dimensions(wsi)

    # L0 size
    w0, h0 = level_dims[0]
    print('[STITCH] original size: {} x {}'.format(w0, h0))

    # --- choose visualization scale, locked to segmentation if virtual ---
    if getattr(wsi_object, "seg_is_virtual", False) and float(getattr(wsi_object, "seg_down", 0)) > 0:
        eff_down = float(wsi_object.seg_down)
        vis_level = -1  # virtual marker
        w_vis = max(1, int(round(w0 / eff_down)))
        h_vis = max(1, int(round(h0 / eff_down)))
        print(f'[STITCH] using VIRTUAL view tied to segmentation: down={eff_down:.2f} → {w_vis} x {h_vis}')
    else:
        vis_level = _safe_get_best_level_for_downsample(wsi, downscale)
        w_vis, h_vis = level_dims[vis_level]
        # compute the effective downscale for logs & later math
        downs = _safe_level_downsamples(wsi)
        dsx, dsy = downs[vis_level]
        eff_down = float((dsx + dsy) * 0.5)
        print('[STITCH] chosen vis_level={}  size: {} x {} (eff_down≈{:.2f})'.format(vis_level, w_vis, h_vis, eff_down))

    # --- guard: PIL pixel limit; auto-increase downscale if needed ---
    pixel_limit = int(getattr(Image, "MAX_IMAGE_PIXELS", 933120000))
    if w_vis * h_vis > pixel_limit:
        factor = math.sqrt((w_vis * h_vis) / float(pixel_limit)) * 1.15  # 15% headroom
        eff_down *= factor
        w_vis = max(1, int(round(w0 / eff_down)))
        h_vis = max(1, int(round(h0 / eff_down)))
        vis_level = -1  # now definitely virtual
        print(f'[STITCH] auto-increased downscale to avoid DecompressionBomb: down={eff_down:.2f} → {w_vis} x {h_vis}')

    # --- read coords & patch params ---
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        slide_name = dset.attrs.get('name', '<unknown>')
        print('[STITCH] start stitching {}'.format(slide_name))
        patch_size = int(dset.attrs['patch_size'])
        patch_level = int(dset.attrs['patch_level'])

    print(f'[STITCH] number of patches for stitching: {len(coords)}')
    print(f'[STITCH] patch_size={patch_size} (target/cnn)  patch_level={patch_level}')

    # Convert target/cnn patch size to native pixels at patch_level
    downs = _safe_level_downsamples(wsi)
    dsx_pl, dsy_pl = downs[patch_level]
    patch_size_native = (int(patch_size * dsx_pl), int(patch_size * dsy_pl))
    print(f'[STITCH] ref patch size (native/L0): {patch_size_native[0]} x {patch_size_native[1]}')

    # --- allocate canvas ---
    from PIL import Image as _Image
    if alpha < 0 or alpha == -1:
        heatmap = _Image.new(size=(w_vis, h_vis), mode="RGB", color=bg_color)
    else:
        heatmap = _Image.new(size=(w_vis, h_vis), mode="RGBA", color=bg_color + (int(255 * alpha),))
    heatmap = np.array(heatmap)

    # --- draw patches ---
    # If vis_level == -1 we’re in a virtual view; pass scale explicitly via wsi_object
    # Convention: DrawMapFromCoords should read wsi_object.vis_is_virtual/vis_down when vis_level == -1
    if not hasattr(wsi_object, "vis_is_virtual"):
        wsi_object.vis_is_virtual = (vis_level == -1)
    if not hasattr(wsi_object, "vis_down"):
        wsi_object.vis_down = eff_down

    heatmap = DrawMapFromCoords(
        heatmap, wsi_object, coords, patch_size_native,
        vis_level, indices=None, draw_grid=draw_grid
    )
    return heatmap

# =============================== Sampling Raw Patches ============================

def SamplePatches(coords_file_path, save_file_path, wsi_object,
                  patch_level=0, custom_downsample=1, patch_size=256,
                  sample_num=100, seed=1, stitch=True, verbose=1, mode='w'):

    with h5py.File(coords_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        h5_patch_size = int(dset.attrs['patch_size'])
        h5_patch_level = int(dset.attrs['patch_level'])

    if verbose > 0:
        print('[SAMPLE] in .h5 file: total patches: {}'.format(len(coords)))
        print('[SAMPLE] in .h5 file: patch size: {}x{}  patch level: {}'.format(h5_patch_size, h5_patch_size, h5_patch_level))

    if patch_level < 0:
        patch_level = h5_patch_level

    if patch_size < 0:
        patch_size = h5_patch_size

    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(coords)), min(len(coords), sample_num), replace=False)

    target_patch_size = np.array([patch_size, patch_size])
    if custom_downsample > 1:
        target_patch_size = (np.array([patch_size, patch_size]) / custom_downsample).astype(np.int32)

    from wsi_core.util_classes import Mosaic_Canvas
    if stitch:
        canvas = Mosaic_Canvas(patch_size=target_patch_size[0], n=sample_num, downscale=4, n_per_row=10, bg_color=(0, 0, 0), alpha=-1)
    else:
        canvas = None

    print(f'[SAMPLE] indices: {indices}')
    wsi = wsi_object.getOpenSlide() if hasattr(wsi_object, 'getOpenSlide') else wsi_object

    for idx in indices:
        coord = coords[idx]
        patch_img = wsi.read_region(tuple(coord), patch_level, tuple([patch_size, patch_size])).convert('RGB')
        if custom_downsample > 1:
            patch_img = patch_img.resize(tuple(target_patch_size))

        if stitch:
            canvas.paste_patch(patch_img)

        asset_dict = {'imgs': np.array(patch_img)[np.newaxis, ...], 'coords': coord}
        save_hdf5(save_file_path, asset_dict, mode=mode)
        mode = 'a'

    return canvas, len(coords), len(indices)
