#!/usr/bin/env python3
"""
Utility: generate and verify heatmap <-> Tiny-ImageNet label reports.

This script replaces ad-hoc CSVs in `reports/` and lets you regenerate:
 - corrected mapping (heatmap folder -> tiny wnid + label)
 - example-derived augmentation (most-common example label per neuron folder)
 - verification summary (train folder exists, heatmap images present)

Usage examples:
  # generate all reports with defaults
  python scripts/generate_heatmap_reports.py --out_dir reports

  # specify dataset roots
  python scripts/generate_heatmap_reports.py --heatmap_root heatmap --examples_root images/example_val_fc \
      --tiny_root datasets/tiny-imagenet-200 --out_dir reports

Outputs (written to --out_dir):
  - heatmap_folder_to_tiny_label.corrected.csv  (canonical folder->wnid,label)
  - heatmap_folder_to_actual_tiny_label.csv     (augmented with example-derived labels)
  - heatmap_folder_verification.csv             (checks: train wnid dir exists, heatmap images present)

"""
import os
import csv
import argparse
import glob
import collections


def load_tiny(tiny_root):
    wnids_path = os.path.join(tiny_root, 'wnids.txt')
    words_path = os.path.join(tiny_root, 'words.txt')
    wnids = []
    words = {}
    if os.path.exists(wnids_path):
        with open(wnids_path, 'r', encoding='utf-8') as f:
            wnids = [l.strip() for l in f if l.strip()]
    else:
        raise FileNotFoundError(f"wnids.txt not found at {wnids_path}")
    if os.path.exists(words_path):
        with open(words_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    words[parts[0]] = parts[1]
    return wnids, words


def canonical_mapping(heatmap_root, wnids, words):
    folders = sorted([d for d in os.listdir(heatmap_root) if os.path.isdir(os.path.join(heatmap_root, d))])
    rows = []
    for folder in folders:
        try:
            idx = int(folder)
        except Exception:
            idx = None
        wnid = ''
        label = ''
        if idx is not None and 0 <= idx < len(wnids):
            wnid = wnids[idx]
            label = words.get(wnid, '')
        rows.append((folder, wnid, label))
    return rows


def augment_with_examples(heatmap_root, examples_root, wnids, words, existing_map):
    # existing_map: dict(folder)->(wnid,label) - may be empty
    folders = [row[0] for row in existing_map]
    augmented = []
    for folder in folders:
        example_dir = os.path.join(examples_root, folder)
        counts = collections.Counter()
        total = 0
        if os.path.isdir(example_dir):
            for p in glob.glob(os.path.join(example_dir, '*')):
                if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.basename(p)
                    parts = name.rsplit('_', 2)
                    lab = None
                    if len(parts) >= 2:
                        last = parts[-1]
                        lab_str = last.split('.')[0]
                        try:
                            lab = int(lab_str)
                        except Exception:
                            lab = None
                    if lab is not None:
                        counts[lab] += 1
                        total += 1
        if counts:
            most_common_lab, most_common_count = counts.most_common(1)[0]
            if 0 <= most_common_lab < len(wnids):
                ex_wnid = wnids[most_common_lab]
                ex_label = words.get(ex_wnid, '')
            else:
                ex_wnid = ''
                ex_label = ''
        else:
            most_common_lab = ''
            most_common_count = 0
            ex_wnid = ''
            ex_label = ''
        prov = next((r for r in existing_map if r[0] == folder), ('', '', ''))
        augmented.append({
            'folder': folder,
            'provided_tiny_wnid': prov[1],
            'provided_tiny_label': prov[2],
            'example_label_idx': str(most_common_lab),
            'example_label_wnid': ex_wnid,
            'example_label_text': ex_label,
            'example_label_count': str(most_common_count),
            'example_total_examples': str(total)
        })
    return augmented


def verify(heatmap_root, tiny_root, wnids):
    train_root = os.path.join(tiny_root, 'train')
    folders = sorted([d for d in os.listdir(heatmap_root) if os.path.isdir(os.path.join(heatmap_root, d))])
    rows = []
    for folder in folders:
        try:
            idx = int(folder)
        except Exception:
            idx = None
        wnid = ''
        train_exists = False
        heatmap_has_images = False
        if idx is not None and 0 <= idx < len(wnids):
            wnid = wnids[idx]
            train_wnid_dir = os.path.join(train_root, wnid)
            train_exists = os.path.isdir(train_wnid_dir)
        # check heatmap images recursively
        hm_path = os.path.join(heatmap_root, folder)
        if os.path.isdir(hm_path):
            for root, _, files in os.walk(hm_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        heatmap_has_images = True
                        break
                if heatmap_has_images:
                    break
        rows.append((folder, idx, wnid, train_exists, heatmap_has_images))
    return rows


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_augmented_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ['folder','provided_tiny_wnid','provided_tiny_label','example_label_idx','example_label_wnid','example_label_text','example_label_count','example_total_examples']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heatmap_root', default='heatmap')
    parser.add_argument('--examples_root', default='images/example_val_fc')
    parser.add_argument('--tiny_root', default='datasets/tiny-imagenet-200')
    parser.add_argument('--out_dir', default='reports')
    args = parser.parse_args()

    wnids, words = load_tiny(args.tiny_root)

    canonical = canonical_mapping(args.heatmap_root, wnids, words)
    corrected_path = os.path.join(args.out_dir, 'heatmap_folder_to_tiny_label.corrected.csv')
    write_csv(corrected_path, ['folder','tiny_wnid','tiny_label'], canonical)

    augmented = augment_with_examples(args.heatmap_root, args.examples_root, wnids, words, canonical)
    augmented_path = os.path.join(args.out_dir, 'heatmap_folder_to_actual_tiny_label.csv')
    write_augmented_csv(augmented_path, augmented)

    verification = verify(args.heatmap_root, args.tiny_root, wnids)
    verification_path = os.path.join(args.out_dir, 'heatmap_folder_verification.csv')
    write_csv(verification_path, ['folder','idx','wnid','train_wnid_exists','heatmap_has_images'], verification)

    print('Wrote files:')
    print(' -', corrected_path)
    print(' -', augmented_path)
    print(' -', verification_path)

if __name__ == '__main__':
    main()
