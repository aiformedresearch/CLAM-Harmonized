#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
Create k stratified patient-level splits for TCGA with an optional
'constrained test' set defined by a CSV of slide_ids. All slides from those
slides' patients are FIXED in TEST across all folds. The remaining patients are
split into train/val/test as usual, but test quotas are reduced per-class to
exclude the constrained patients.

What you get
------------
- Standard outputs (unchanged interface):
  splits/<task>_<pct>/splits_i.csv
  splits/<task>_<pct>/splits_i_bool.csv
  splits/<task>_<pct>/splits_i_descriptor.csv
- NEW separate views for test:
  splits_i_test_constrained_patients.csv / slides.csv
  splits_i_test_regular_patients.csv     / slides.csv

Logging
-------
Prints per-class distributions (patients & slides) for:
  - train
  - val
  - test (union of constrained + regular)
  - test_constrained (forced)

Also prints a clear note that TEST = TEST_REGULAR + TEST_FORCED.
"""

import os
import argparse
import numpy as np
import pandas as pd

from datasets.dataset_mtl_concat import Generic_WSI_MTL_Dataset, Generic_MIL_MTL_Dataset, save_splits

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default=-1, help='fraction of labels (default: [1.0])')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of splits (default: 10)')
parser.add_argument('--hold_out_test', action='store_true', default=False, help='hold out test set (default: False)')
parser.add_argument('--split_code', type=str, default=None)
parser.add_argument('--task', type=str, choices=['TCGA'])
parser.add_argument('--force_test_slides_csv', type=str, default=None,
                    help='CSV with a column "slide_id" (or first column) whose patients are FIXED in TEST across folds')
args = parser.parse_args()


# ----------------------------
# Helpers (dataset-agnostic)
# ----------------------------
def _dataset_df(dataset):
    for cand in ['slide_data', 'slides_df', 'df', 'data', 'metadata']:
        if hasattr(dataset, cand):
            obj = getattr(dataset, cand)
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
    raise RuntimeError("Dataset dataframe not found (tried slide_data/slides_df/df/data/metadata).")


def _find_col(df, candidates):
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None


def _build_maps(dataset):
    df = _dataset_df(dataset)
    slide_col = _find_col(df, {'slide_id', 'slide', 'wsi', 'filename'})
    pat_col   = _find_col(df, {'patient', 'patient_id', 'case_id', 'case', 'subject'})
    cls_col   = _find_col(df, {'label', 'primary_label', 'y', 'class', 'site_primary', 'site'})

    if slide_col is None or pat_col is None:
        raise RuntimeError("Expected slide_id and patient columns in dataset dataframe.")

    # class name -> id
    class_to_id = {}
    if hasattr(dataset, 'label_dicts') and dataset.label_dicts and isinstance(dataset.label_dicts[0], dict):
        class_to_id = dataset.label_dicts[0]
    id_to_class = {v: k for k, v in class_to_id.items()}

    # class id per row
    if cls_col is not None and class_to_id:
        if df[cls_col].dtype.kind in 'iu':
            df['_class_id'] = df[cls_col].astype(int)
        else:
            df['_class_id'] = df[cls_col].map(class_to_id)
    else:
        df['_class_id'] = np.nan

    # maps
    slide_to_patient = df.set_index(slide_col)[pat_col].astype(str).to_dict()
    patient_to_slides = df.groupby(pat_col)[slide_col].apply(list).to_dict()

    def _pat_cls(s):
        s = s.dropna()
        return int(s.mode().iloc[0]) if len(s) else -1
    patient_to_class = df.groupby(pat_col)['_class_id'].apply(_pat_cls).to_dict()

    return {
        'df': df,
        'slide_col': slide_col,
        'pat_col': pat_col,
        'slide_to_patient': slide_to_patient,
        'patient_to_slides': patient_to_slides,
        'patient_to_class': patient_to_class,
        'id_to_class': id_to_class
    }


def _load_forced_patients(force_csv_path, maps):
    forced_df = pd.read_csv(force_csv_path)
    if 'slide_id' not in forced_df.columns:
        forced_df['slide_id'] = forced_df.iloc[:, 0]
    slides = set(map(str, forced_df['slide_id'].astype(str).tolist()))
    slide_to_patient = maps['slide_to_patient']
    patients = {slide_to_patient[s] for s in slides if s in slide_to_patient}
    return patients


def _count_per_class(patients, patient_to_class, n_classes):
    counts = np.zeros(n_classes, dtype=int)
    for p in patients:
        cid = patient_to_class.get(p, -1)
        if isinstance(cid, (int, np.integer)) and 0 <= cid < n_classes:
            counts[cid] += 1
    return counts


def _count_slides(patients, patient_to_slides):
    return sum(len(patient_to_slides.get(p, [])) for p in patients)


def _print_split_distributions(title, splits_patients, maps, n_classes):
    """
    splits_patients: dict split_name -> set(patient_ids)
    Prints per-class patient counts and slide counts for each split.
    """
    id2c = maps['id_to_class']
    ptc = maps['patient_to_class']
    pts = maps['patient_to_slides']

    print(f"\n=== {title} ===")
    # header line
    header = ["Split"]
    header += [f"{id2c.get(c, str(c))}" for c in range(n_classes)]
    header += ["TOTAL_patients", "TOTAL_slides"]
    print("\t".join(header))

    for split_name, pset in splits_patients.items():
        per_cls = _count_per_class(pset, ptc, n_classes)
        total_pat = int(per_cls.sum())
        total_sld = _count_slides(pset, pts)
        row = [split_name] + [str(per_cls[c]) for c in range(n_classes)] + [str(total_pat), str(total_sld)]
        print("\t".join(row))


# ----------------------------
# Build dataset
# ----------------------------
if args.task == 'TCGA':
    args.n_classes = 18
    dataset = Generic_WSI_MTL_Dataset(
        csv_path='dataset_csv/TCGA.csv',
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dicts=[
            {
                "Adrenal": 0, "Bladder": 1, "Breast": 2, "Cervix": 3, "Cholangio": 4,
                "Endometrial": 5, "Gastrointestinal": 6, "Germ cell": 7, "Glioma": 8,
                "Head and Neck": 9, "Liver": 10, "Lung": 11, "Ovarian": 12, "Pancreatic": 13,
                "Prostate": 14, "Renal": 15, "Skin": 16, "Thyroid": 17
            },
            {},  # keep order to match downstream scripts
            {'F': 0, 'M': 1}
        ],
        label_cols=['label', 'site', 'sex'],
        patient_strat=True
    )
else:
    print(f'task {args.task} not defined')
    raise NotImplementedError

# Per-class patient counts (for quotas)
num_patients_per_class = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.floor(num_patients_per_class * 0.10).astype(int)
test_num = np.floor(num_patients_per_class * 0.20).astype(int)
print("val_num per class:", val_num)
print("test_num per class:", test_num)

# Label fractions
if args.label_frac > 0:
    label_fracs = [args.label_frac]
else:
    label_fracs = [1.0]

# ----------------------------
# Constrained test patients BEFORE splitting
# ----------------------------
forced_test_patients = set()
test_num_adj = test_num.copy()

maps = None
if args.force_test_slides_csv:
    maps = _build_maps(dataset)
    forced_test_patients = _load_forced_patients(args.force_test_slides_csv, maps)

    # Diagnostics: forced per class (patients/slides) and remaining pool
    forced_per_class_pat = _count_per_class(forced_test_patients, maps['patient_to_class'], args.n_classes)
    pts_map = maps['patient_to_slides']
    ptc_map = maps['patient_to_class']
    forced_slides_per_class = np.zeros(args.n_classes, dtype=int)
    for p in forced_test_patients:
        c = ptc_map.get(p, -1)
        if 0 <= c < args.n_classes:
            forced_slides_per_class[c] += len(pts_map.get(p, []))

    total_patients_per_class = np.array([len(cls) for cls in dataset.patient_cls_ids], dtype=int)
    remaining_patients_per_class = total_patients_per_class - forced_per_class_pat

    # Adjust test quotas
    test_num_adj = np.maximum(test_num - forced_per_class_pat, 0)

    forced_pat = len(forced_test_patients)
    forced_sld = _count_slides(forced_test_patients, maps['patient_to_slides'])
    print("\n[Constrained TEST] summary:")
    print(f"  Forced patients: {forced_pat}")
    print(f"  Forced slides:   {forced_sld}")
    print("  Per-class forced (patients):", forced_per_class_pat.tolist())
    print("  Per-class forced (slides):  ", forced_slides_per_class.tolist())
    print("  Remaining pool (patients):  ", remaining_patients_per_class.tolist())
    print("  Adjusted remaining test quotas:", test_num_adj.tolist())

# If user also asked for hold-out test (without constrained): keep behavior.
if args.hold_out_test and not args.force_test_slides_csv:
    custom_test_ids_global = dataset.sample_held_out(test_num=test_num)
else:
    # IMPORTANT: custom_test_ids_global is a list of PATIENT IDs (strings)
    custom_test_ids_global = list(forced_test_patients) if len(forced_test_patients) > 0 else None

# ----------------------------
# Create splits and save
# ----------------------------
for lf in label_fracs:
    if args.split_code is not None:
        split_dir = 'splits/' + str(args.split_code) + '_{}'.format(int(lf * 100))
    else:
        split_dir = 'splits/' + str(args.task) + '_{}'.format(int(lf * 100))
    os.makedirs(split_dir, exist_ok=True)

    # IMPORTANT: pass adjusted test quotas + custom constrained patients
    dataset.create_splits(
        k=args.k,
        val_num=val_num,
        test_num=test_num_adj if args.force_test_slides_csv else test_num,
        label_frac=lf,
        custom_test_ids=custom_test_ids_global
    )

    for i in range(args.k):
        print(f"\n==== info about split {i} ====")
        if dataset.split_gen is None:
            print('using get_split_from_df function')
            ids = []
            for split in ['train', 'val', 'test']:
                ids.append(dataset.get_split_from_df(pd.read_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i))),
                                                     split_key=split, return_ids_only=True))
            dataset.train_ids = ids[0]
            dataset.val_ids = ids[1]
            dataset.test_ids = ids[2]
        else:
            print('using set_splits function')
            dataset.set_splits()

        # --- Build patient-ID sets for each split ---
        if maps is None:
            maps = _build_maps(dataset)

        # Convert current slide indices -> patient IDs (strings)
        train_pat_ids = set(dataset.slide_data.loc[dataset.train_ids, 'case_id'].astype(str))
        val_pat_ids   = set(dataset.slide_data.loc[dataset.val_ids,   'case_id'].astype(str))
        test_pat_ids  = set(dataset.slide_data.loc[dataset.test_ids,  'case_id'].astype(str))

        # --- Compute constrained vs regular in PATIENT space (FIX) ---
        if args.force_test_slides_csv and custom_test_ids_global is not None:
            forced_patients_global = set(map(str, custom_test_ids_global))  # patient IDs
            test_constrained_patients = test_pat_ids & forced_patients_global
            test_regular_patients     = test_pat_ids - test_constrained_patients
        else:
            test_constrained_patients = set()
            test_regular_patients     = test_pat_ids

        # --------- LOG distributions (patients & slides) ----------
        print("\nNOTE: In all tables below, TEST = TEST_REGULAR + TEST_FORCED (aka TEST_CONSTRAINED).")
        split_patient_sets = {
            'train': train_pat_ids,
            'val': val_pat_ids,
            'test': test_pat_ids,                         # union view
            'test_constrained': test_constrained_patients # constrained-only view
        }
        _print_split_distributions("Per-class distribution (patients/slides)", split_patient_sets, maps, args.n_classes)

        # --------- Save descriptor and splits ----------
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        # Print the FULL table (not just head), so every class row is visible
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print("saving splits descriptor:\n", descriptor_df.to_string(index=True))
        descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)), index=False)

        # --------- Save classic splits ----------
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)

        # --------- Save constrained vs regular test (patient-wise) ----------
        pd.Series(sorted(test_constrained_patients), name='patient_id').to_csv(
            os.path.join(split_dir, f'splits_{i}_test_constrained_patients.csv'), index=False
        )
        pd.Series(sorted(test_regular_patients), name='patient_id').to_csv(
            os.path.join(split_dir, f'splits_{i}_test_regular_patients.csv'), index=False
        )

        # --------- Also save slide-wise lists ----------
        try:
            pts = maps['patient_to_slides']
            cons_slides = [s for p in test_constrained_patients for s in pts.get(p, [])]
            reg_slides  = [s for p in test_regular_patients  for s in pts.get(p, [])]
            pd.Series(sorted(cons_slides), name='slide_id').to_csv(
                os.path.join(split_dir, f'splits_{i}_test_constrained_slides.csv'), index=False
            )
            pd.Series(sorted(reg_slides), name='slide_id').to_csv(
                os.path.join(split_dir, f'splits_{i}_test_regular_slides.csv'), index=False
            )
        except Exception as e:
            print(f"[warn] could not save slide-wise constrained/regular files: {e}")

        print(f"[Fold {i}] TEST union(slides)={len(dataset.test_ids)} "
              f"| constrained(pat)={len(test_constrained_patients)} "
              f"| regular(pat)={len(test_regular_patients)}")
    print(f"\n[Done] splits saved in {split_dir}/")