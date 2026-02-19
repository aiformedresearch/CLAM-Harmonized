from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset, save_splits
import h5py
from utils.eval_utils_mtl_concat import *
import time

# Training settings
parser = argparse.ArgumentParser(description='TOAD Evaluation Script')
parser.add_argument('--data_root_dir', type=str, help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. ' +
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints)')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_average for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['TCGA'])

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

print('results folder content:', os.listdir(args.results_dir))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

print('looking for model in path:', args.models_dir)
assert os.path.isdir(args.models_dir)

print('looking for splits in dir:', args.splits_dir)
assert os.path.isdir(args.splits_dir)

settings = {
    'task': args.task,
    'split': args.split,
    'save_dir': args.save_dir, 
    'models_dir': args.models_dir,
    'drop_out': args.drop_out,
    'micro_avg': args.micro_average
}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)

print(settings)


if args.task == 'TCGA':
    args.n_classes = 18
    dataset = Generic_MIL_MTL_Dataset(
        csv_path='dataset_csv/TCGA.csv',
        data_dir=os.path.join(args.data_root_dir, 'pt_files'),
        shuffle=False,
        print_info=True,
        label_dicts = [
                        {
                            "Adrenal": 0, "Bladder": 1, "Breast": 2, "Cervix": 3, "Cholangio": 4,
                            "Endometrial": 5, "Gastrointestinal": 6, "Germ cell": 7, "Glioma": 8,
                            "Head and Neck": 9, "Liver": 10, "Lung": 11, "Ovarian": 12, "Pancreatic": 13,
                            "Prostate": 14, "Renal": 15, "Skin": 16, "Thyroid": 17
                        },
                      {}, # better to leave this order and content to not messup the dataset_mtl_contat.py script
                      {'F': 0, 'M': 1}],
        label_cols=['label', 'site', 'sex'], # better to leave this order and content to not messup the dataset_mtl_contat.py script
        patient_strat=True)

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
args.label_dicts = dataset.label_dicts
print(f'labels dict: {args.label_dicts}')


def check_data_leakage(datasets, datasets_id):
    """
    Checks for data leakage across train, validation, and test sets.
    
    Parameters:
    - datasets: A dictionary containing datasets for 'train', 'val', and 'test'.
    - datasets_id: A dictionary mapping 'train', 'val', and 'test' to their respective indices.
    
    Returns:
    - None: Prints the result of the leakage check.
    """
    # Extract case_ids for each split
    train_case_ids = set(datasets[datasets_id['train']].slide_data['case_id'])
    val_case_ids = set(datasets[datasets_id['val']].slide_data['case_id'])
    test_case_ids = set(datasets[datasets_id['test']].slide_data['case_id'])
    
    # Check for overlaps between the sets
    train_val_overlap = train_case_ids.intersection(val_case_ids)
    train_test_overlap = train_case_ids.intersection(test_case_ids)
    val_test_overlap = val_case_ids.intersection(test_case_ids)
    
    # Report the results
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("No data leakage detected between train, validation, and test sets.")
    else:
        print("Data leakage detected!")
        if train_val_overlap:
            print(f"Overlap between TRAIN and VAL sets: {train_val_overlap}")
        if train_test_overlap:
            print(f"Overlap between TRAIN and TEST sets: {train_test_overlap}")
        if val_test_overlap:
            print(f"Overlap between VAL and TEST sets: {val_test_overlap}")


if __name__ == "__main__":

    print('start eval_mtl_concat')
    start_eval = time.time()

    # Clear cache and reset tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    all_cls_auc = []
    all_cls_acc = []
    all_cls_top3_acc = []
    all_cls_top5_acc = []


    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
            csv_path = None
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        print("some general info and checking")
        print(f"TRAIN set num slides with possible repetition: {len(datasets[datasets_id['train']].slide_data['slide_id'])}")
        print(f"TRAIN set num slides set: {len(set(datasets[datasets_id['train']].slide_data['slide_id']))}")
        print(f"TRAIN set num patients with possible repetitions: {len(datasets[datasets_id['train']].slide_data['case_id'])}")
        print(f"TRAIN set num patients set: {len(set(datasets[datasets_id['train']].slide_data['case_id']))}")

        print(f"VAL set num slides with possible repetition: {len(datasets[datasets_id['val']].slide_data['slide_id'])}")
        print(f"VAL set num slides set: {len(set(datasets[datasets_id['val']].slide_data['slide_id']))}")
        print(f"VAL set num patients with possible repetitions: {len(datasets[datasets_id['val']].slide_data['case_id'])}")
        print(f"VAL set num patients set: {len(set(datasets[datasets_id['val']].slide_data['case_id']))}")

        print(f"TEST set num slides with possible repetition: {len(datasets[datasets_id['test']].slide_data['slide_id'])}")
        print(f"TEST set num slides set: {len(set(datasets[datasets_id['test']].slide_data['slide_id']))}")
        print(f"TEST set num patients with possible repetitions: {len(datasets[datasets_id['test']].slide_data['case_id'])}")
        print(f"TEST set num patients set: {len(set(datasets[datasets_id['test']].slide_data['case_id']))}")
        
        print(f"some examples of slide ids: {datasets[datasets_id['train']].slide_data['slide_id'][0:5]}")
        print(f"some examples of patient ids: {datasets[datasets_id['train']].slide_data['case_id'][0:5]}")

        # Example usage:
        check_data_leakage(datasets, datasets_id)


        # Print which split is being used and its size
        print(f"FOR EVALUATION, USING SPLIT: {args.split}")
        print(f"Split size: {len(split_dataset)}")  # Assuming split_dataset has a __len__ method

        model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        for cls_idx in range(len(results_dict['cls_aucs'])):
            print('class {} auc: {}'.format(cls_idx, results_dict['cls_aucs'][cls_idx]))

        all_cls_auc.append(results_dict['cls_auc'])
        all_cls_acc.append(1-results_dict['cls_test_error'])
        all_cls_top3_acc.append(results_dict['top3_acc'])
        all_cls_top5_acc.append(results_dict['top5_acc'])
        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)


    df_dict = {'folds': folds, 'cls_test_auc': all_cls_auc, 'cls_test_acc': all_cls_acc, 'cls_top3_acc': all_cls_top3_acc, 'cls_top5_acc': all_cls_top5_acc}

    final_df = pd.DataFrame(df_dict)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

    time_difference = time.time() - start_eval
    hours, remainder = divmod(time_difference, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time to perform evaluation: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    # Get the maximum memory allocated by tensors
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

    # Get the maximum memory reserved by the caching allocator
    max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # Convert to MB

    print(f"Maximum Memory Allocated by Tensors: {max_memory_allocated:.2f} MB")
    print(f"Maximum Memory Reserved by Caching Allocator: {max_memory_reserved:.2f} MB")
