# internal imports
from wsi_core.WholeSlideImage_harmonize import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import sys
import numpy as np
import time
import argparse
import pandas as pd
from tqdm import tqdm
import math
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg") # prevents any GUI backend
import os


# ------------------------------ helpers --------------------------------------

def stitching(file_path, wsi_object, downscale=32):
    t0 = time.time()
    # Use the same (virtual) downscale factor chosen for segmentation if present
    eff_downscale = wsi_object.seg_down if getattr(wsi_object, 'seg_is_virtual', False) else downscale
    heatmap = StitchCoords(file_path, wsi_object, downscale=eff_downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    return heatmap, time.time() - t0

def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    t0 = time.time()
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    return WSI_object, time.time() - t0

def patching(WSI_object, **kwargs):
    t0 = time.time()
    file_path = WSI_object.process_contours(**kwargs)
    return file_path, time.time() - t0

def _best_level_for_downsample(level_downsamples, target_ds=32.0):
    if not level_downsamples or len(level_downsamples) == 0:
        return 0
    ds_iso = [float((d[0] + d[1]) * 0.5) for d in level_downsamples]
    return int(np.argmin([abs(d - target_ds) for d in ds_iso]))

# ----------------------- main processing routine -----------------------------

def seg_and_patch(
    source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
    patch_size=256, step_size=256,
    seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                'keep_ids': 'none', 'exclude_ids': 'none'},
    filter_params={'a_t':100, 'a_h': 16, 'max_n_holes':8},
    vis_params={'vis_level': -1, 'line_thickness': 250},
    patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
    patch_level=0,
    use_default_params=False,
    seg=False, save_mask=True,
    stitch=False,
    patch=False, auto_skip=True, process_list=None,
    harmonize=False, target_mpp=0.25,
    qa_viz=True,
    desired_seg_downsample=32.0,
    max_eff_pixels=300_000_000
):
    print('source', source, flush=True)
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    print('slides:', slides, flush=True)

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)
    print(df, flush=True)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled', flush=True)
        df = df.assign(**{
            'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
            'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
            'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
            'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
            'contour_fn': np.full((len(df)), patch_params['contour_fn'])
        })

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    save_dir_metadata = os.path.join(save_dir, 'metadata')
    os.makedirs(save_dir_metadata, exist_ok=True)

    DEFAULT_VIEW_DS = 32.0

    for i in tqdm(range(total), mininterval=0.5):
        idx = process_stack.index[i]
        slide_fn = process_stack.loc[idx, 'slide_id']
        print(f"\n\nprogress: {i/total:.2f}, {i}/{total}", flush=True)
        print(f'processing {slide_fn}', flush=True)

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide_fn)

        # if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
        #     print(f'{slide_id} already exist in destination location, skipped', flush=True)
        #     df.loc[idx, 'status'] = 'already_exist'
        #     # write status after each slide to keep UI responsive
        #     df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        #     continue

        # ---------------- WSI OPEN ----------------
        full_path = os.path.join(source, slide_fn)
        print(f"[WSI] Opening: {full_path}", flush=True)
        t_open = time.time()
        try:
            WSI_object = WholeSlideImage(full_path)
        except Exception as e:
            print(f"[ERROR] Failed to open slide: {e}", flush=True)
            df.loc[idx, 'status'] = 'failed_open'
            df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
            continue
        print(f"[WSI] Opened in {time.time()-t_open:.2f}s", flush=True)

        # ---------------- HARMONIZATION ----------------
        WSI_object.set_harmonization(
            harmonize=bool(harmonize),
            target_mpp=float(target_mpp),
            desired_seg_downsample=float(desired_seg_downsample)
        )

        # NEW: Hard-skip this slide when MPP is missing
        if bool(harmonize) and getattr(WSI_object, "harmonization_skipped", False):
            print("[SKIP] Missing MPP metadata; skipping slide entirely.")
            df.loc[idx, 'status'] = 'skipped_missing_mpp'
            df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
            continue


        # Save slide metadata
        meta_blob = {
            "slide": slide_fn,
            "mpp_x": WSI_object.mpp_x,
            "mpp_y": WSI_object.mpp_y,
            "objective": WSI_object.objective,
            "levels": len(WSI_object.level_dim),
            "level_downsamples": WSI_object.level_downsamples,
            "harmonize": bool(harmonize),
            "target_mpp": float(target_mpp),
            "dx_target": WSI_object.dx_target,
            "dy_target": WSI_object.dy_target,
            "seg_level": WSI_object.seg_level,
            "seg_is_virtual": WSI_object.seg_is_virtual,
            "seg_down": WSI_object.seg_down,
            "dx_seg": WSI_object.dx_seg,
            "dy_seg": WSI_object.dy_seg,
            "patch_size_native_scalar": int(patch_size),
            "desired_seg_downsample": float(desired_seg_downsample)
        }
        meta_json = os.path.join(save_dir_metadata, f"{slide_id}_meta.json")
        with open(meta_json, "w") as f:
            json.dump(meta_blob, f, indent=2)
        print(f"[DBG] wrote metadata → {meta_json}", flush=True)

        if qa_viz and bool(harmonize):
            qa_out = os.path.join(save_dir_metadata, f"{slide_id}_harmonization_viz_target.jpg")
            print("[QA] generating harmonization viz…", flush=True)
            t_qa = time.time()
            WSI_object.save_harmonization_viz(qa_out, visualization_patch_size=patch_size, compare_mode="target")
            print(f"[QA] viz saved in {time.time()-t_qa:.2f}s", flush=True)

        # ---------------- parameter sets ----------------
        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}
            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})
            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level_tmp = df.loc[idx, 'seg_level']
                    seg_level_tmp = int(seg_level_tmp) if seg_level_tmp >= 0 else WSI_object.seg_level
                    scale = WSI_object.level_downsamples[seg_level_tmp] if not WSI_object.seg_is_virtual else (WSI_object.seg_down, WSI_object.seg_down)
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})
            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})
            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        # ---------------- pick vis/seg levels ----------------
        if int(current_vis_params['vis_level']) < 0:
            if len(WSI_object.level_dim) == 1:
                # OLD: print("[VIS] Single level; using 0")
                # NEW: use a virtual view ~32x down like segmentation
                virtual_view_down = float(getattr(WSI_object, "seg_down", DEFAULT_VIEW_DS)) or DEFAULT_VIEW_DS
                current_vis_params['vis_level'] = -1  # signal "virtual"
                # expose a hint for visWSI/StitchCoords
                WSI_object.vis_is_virtual = True
                WSI_object.vis_down = virtual_view_down
                print(f"[VIS] Single level; using VIRTUAL view down={virtual_view_down:.1f}", flush=True)
            else:
                best_level = _best_level_for_downsample(WSI_object.level_downsamples, DEFAULT_VIEW_DS)
                print(f"[VIS] picked best level for ~{DEFAULT_VIEW_DS}× downsample → {best_level}", flush=True)
                current_vis_params['vis_level'] = best_level

        if int(current_seg_params['seg_level']) < 0:
            print(f"[SEG] using chosen seg_level from harmonization plan → {WSI_object.seg_level}", flush=True)
            current_seg_params['seg_level'] = int(WSI_object.seg_level)

        # normalize keep/exclude
        keep_ids = str(current_seg_params['keep_ids'])
        current_seg_params['keep_ids'] = [] if keep_ids == 'none' or len(keep_ids) == 0 else np.array(keep_ids.split(',')).astype(int)
        exclude_ids = str(current_seg_params['exclude_ids'])
        current_seg_params['exclude_ids'] = [] if exclude_ids == 'none' or len(exclude_ids) == 0 else np.array(exclude_ids.split(',')).astype(int)

        # ---------------- effective segmentation canvas guard ----------------
        W0, H0 = WSI_object.level_dim[0]
        if WSI_object.seg_is_virtual:
            eff_w = max(1, int(math.floor(W0 / WSI_object.seg_down)))
            eff_h = max(1, int(math.floor(H0 / WSI_object.seg_down)))
            eff_desc = f"virtual down={WSI_object.seg_down:.1f} → {eff_w}x{eff_h}"
        else:
            lvl = int(current_seg_params['seg_level'])
            eff_w, eff_h = WSI_object.level_dim[lvl]
            eff_desc = f"native level {lvl} → {eff_w}x{eff_h}"
        print(f"[SEG] effective segmentation canvas: {eff_desc}", flush=True)

        if eff_w * eff_h > int(max_eff_pixels):
            print(f"effective seg canvas {eff_w}x{eff_h} ({eff_w*eff_h/1e6:.1f} MP) exceeds limit ({int(max_eff_pixels)/1e6:.1f} MP); skipping segmentation", flush=True)
            df.loc[idx, 'status'] = 'failed_seg'
            df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
            continue

        df.loc[idx, 'vis_level'] = int(current_vis_params['vis_level'])
        df.loc[idx, 'seg_level'] = int(current_seg_params['seg_level'])

        # ---------------- run stages ----------------
        seg_time_elapsed = -1
        if seg:
            print('starting segment', flush=True)
            t_seg = time.time()
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
            print(f'segment done in {time.time()-t_seg:.2f}s', flush=True)

        if save_mask:
            print('starting visWSI = masking', flush=True)
            t_mask = time.time()
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)
            print(f'done visWSI in {time.time()-t_mask:.2f}s', flush=True)

        patch_time_elapsed = -1
        if patch:
            print('starting patching', flush=True)
            current_patch_params.update({
                'patch_level': patch_level,
                'patch_size': int(patch_size),
                'step_size': int(step_size),
                'save_path': patch_save_dir
            })
            t_patch = time.time()
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params)
            print(f'patching done in {time.time()-t_patch:.2f}s', flush=True)

        stitch_time_elapsed = -1
        if stitch:
            print('stitching started', flush=True)
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                t_stitch = time.time()
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=32)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)
                print(f'stitching done in {time.time()-t_stitch:.2f}s', flush=True)
            else:
                print('stitching skipped (no .h5)', flush=True)

        print(f"segmentation took {seg_time_elapsed} seconds", flush=True)
        print(f"patching took {patch_time_elapsed} seconds", flush=True)
        print(f"stitching took {stitch_time_elapsed} seconds", flush=True)
        df.loc[idx, 'status'] = 'processed'

        # persist progress after each slide (so you see movement even if the next one is slow)
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

        seg_times += seg_time_elapsed if seg_time_elapsed >= 0 else 0.0
        patch_times += patch_time_elapsed if patch_time_elapsed >= 0 else 0.0
        stitch_times += stitch_time_elapsed if stitch_time_elapsed >= 0 else 0.0

    denom = max(1, total)
    seg_times /= denom
    patch_times /= denom
    stitch_times /= denom

    # final write
    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times), flush=True)
    print("average patching time in s per slide: {}".format(patch_times), flush=True)
    print("average stiching time in s per slide: {}".format(stitch_times), flush=True)
    return seg_times, patch_times

# --------------------------------- CLI ---------------------------------------

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type=int, default=256, help='step_size')
parser.add_argument('--patch_size', type=int, default=448, help='patch_size (defined at target_mpp when --harmonize)')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str, help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str, help='predefined profile (.csv)')
parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
parser.add_argument('--process_list', type=str, default=None, help='name of list (.csv)')
parser.add_argument('--harmonize', default=False, action='store_true', help='use MPP metadata to harmonize to target_mpp')
parser.add_argument('--target_mpp', type=float, default=0.25, help='target microns-per-pixel for harmonization (default 0.25)')
parser.add_argument('--qa_viz', default=True, help='saving harmonization quality-assurance image')
parser.add_argument('--desired_seg_downsample', type=float, default=32.0, help='target downsample for segmentation canvas (virtual if needed)')
parser.add_argument('--max_eff_pixels', type=int, default=300_000_000, help='max effective segmentation pixels (proxy/native) to allow')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    process_list = os.path.join(args.save_dir, args.process_list) if args.process_list else None

    print('source: ', args.source, flush=True)
    print('patch_save_dir: ', patch_save_dir, flush=True)
    print('mask_save_dir: ', mask_save_dir, flush=True)
    print('stitch_save_dir: ', stitch_save_dir, flush=True)

    directories = {
        'source': args.source,
        'save_dir': args.save_dir,
        'patch_save_dir': patch_save_dir,
        'mask_save_dir': mask_save_dir,
        'stitch_save_dir': stitch_save_dir
    }

    for key, val in directories.items():
        print(f"{key} : {val}", flush=True)
        if key != 'source':
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':16, 'a_h': 4, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 100}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]
        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]
        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]
        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {
        'seg_params': seg_params,
        'filter_params': filter_params,
        'patch_params': patch_params,
        'vis_params': vis_params
    }

    print('###### input run parameters: \n', parameters, flush=True)

    seg_times, patch_times = seg_and_patch(
        **directories, **parameters,
        patch_size=args.patch_size, step_size=args.step_size,
        seg=args.seg, use_default_params=False, save_mask=True,
        stitch=args.stitch, patch_level=args.patch_level, patch=args.patch,
        process_list=process_list, auto_skip=args.no_auto_skip,
        harmonize=args.harmonize, target_mpp=args.target_mpp,
        qa_viz=args.qa_viz,
        desired_seg_downsample=args.desired_seg_downsample,
        max_eff_pixels=args.max_eff_pixels
    )
