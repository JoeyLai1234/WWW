#!/usr/bin/env python3
"""
reimpl.py

Minimal, clean reimplementation of the WWW concept-matching pipeline.

This script provides a compact, readable pipeline that:
 - loads Tiny-ImageNet labels and applies a CLIP prompt template
 - computes CLIP text embeddings (batched) with optional caching
 - loads precomputed neuron/image vectors (from `utils/img_features_{num_example}.pkl`)
 - computes cosine similarities and selects top-k words per neuron
 - saves predictions pickle (list-of-lists) compatible with `compute_score.py`
 - optionally generates heatmaps. Two modes:
     * top-word heatmaps (word similarity bars) -- legacy
     * spatial (patch-level) heatmaps over images using CLIP image encoder
 - optional MPNet re-ranking of candidate word lists and MPNet-based combined heatmaps

Notes / assumptions:
 - Spatial heatmaps compute CLIP embeddings for sliding-window image patches
   and score each patch by cosine similarity with the neuron vector (slow but simple).
 - MPNet is used only on text: we build a small MPNet "prototype" for a neuron by
   averaging MPNet embeddings of its top-CLIP words and use that to re-rank candidates.

Run example (from repo root):
    .venv/bin/python scripts/reimpl.py --num_neurons 50 --num_words 500 --top_k 5 --num_example 8 --heatmaps --spatial

"""
from pathlib import Path
import argparse
import pickle
import json
import numpy as np
import sys
import subprocess
import math
from tqdm import tqdm

try:
    import clip
    import torch
except Exception:
    clip = None
    torch = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def parse_args():
    p = argparse.ArgumentParser(description='reimplementation of WWW concept-matching (minimal)')
    p.add_argument('--num_neurons', type=int, default=50)
    p.add_argument('--num_words', type=int, default=1000)
    p.add_argument('--num_example', type=int, default=8)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--template', default='A photo of a {label}', help='Simple CLIP template')
    p.add_argument('--cache', default='utils/text_emb_cache.pkl', help='Where to cache CLIP text embeddings')
    p.add_argument('--mpnet_cache', default='utils/mpnet_text_emb_cache.pkl', help='Where to cache MPNet text embeddings')
    p.add_argument('--out', default='utils/www_reimpl_preds.pkl', help='Predictions pickle output')
    p.add_argument('--eval', action='store_true', help='Run compute_score after generation')
    p.add_argument('--mpnet', action='store_true', help='Compute MPNet re-ranking (requires sentence-transformers)')
    p.add_argument('--mpnet_alpha', type=float, default=0.5, help='Weight for CLIP vs MPNet in combined score (0..1)')
    p.add_argument('--heatmaps', action='store_true', help='Generate heatmaps (legacy top-word or spatial)')
    p.add_argument('--spatial', action='store_true', help='Generate spatial (patch-level) heatmaps over images using CLIP')
    p.add_argument('--heatmap_top', type=int, default=40, help='Number of top words to include in top-word heatmap')
    p.add_argument('--device', default='cpu', help='torch device to use for CLIP/MPNet (cpu or cuda)')
    return p.parse_args()


def load_tiny_words(max_words):
    path = Path('datasets') / 'tiny-imagenet-200' / 'words.txt'
    words = []
    if not path.exists():
        raise FileNotFoundError(f"Tiny words file not found at {path}. Please add Tiny-ImageNet or change --num_words to a smaller value.")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wnid, label = parts[0], parts[1]
                words.append(label)
            if len(words) >= max_words:
                break
    return words


def compute_clip_text_embeddings(words, template, cache_path, device='cpu', batch=128):
    """Compute (or load from cache) CLIP text embeddings for templated words.

    Returns: dict mapping label -> numpy vector (L2-normalized)
    """
    cache_p = Path(cache_path)
    if cache_p.exists():
        try:
            data = pickle.load(open(cache_p, 'rb'))
            if set(data.keys()) >= set(words):
                print('Loaded text embeddings from cache:', cache_p)
                return data
        except Exception:
            print('Failed to load cache; recomputing')

    if clip is None or torch is None:
        raise RuntimeError('clip or torch not available in the environment; install dependencies or run inside the repo venv')

    model, _ = clip.load('ViT-B/16', device=device)
    mapped = {}
    templated = [template.format(label=w) for w in words]
    for i in range(0, len(templated), batch):
        chunk_labels = words[i:i+batch]
        chunk_texts = templated[i:i+batch]
        toks = clip.tokenize(chunk_texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(toks).cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-10, None)
        for label, vec in zip(chunk_labels, emb):
            mapped[label] = vec

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(mapped, open(cache_p, 'wb'))
    print('Wrote text embedding cache to', cache_p)
    return mapped


def load_mpnet_text_embeddings(words, cache_path, device='cpu', batch=256):
    cache_p = Path(cache_path)
    if cache_p.exists():
        try:
            data = pickle.load(open(cache_p, 'rb'))
            if set(data.keys()) >= set(words):
                print('Loaded MPNet text embeddings from cache:', cache_p)
                return data
        except Exception:
            print('Failed to load MPNet cache; recomputing')

    if SentenceTransformer is None:
        raise RuntimeError('sentence_transformers not available; install sentence-transformers to use --mpnet')

    model = SentenceTransformer('all-mpnet-base-v2')
    mapped = {}
    for i in range(0, len(words), batch):
        chunk = words[i:i+batch]
        emb = model.encode(chunk, show_progress_bar=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-10, None)
        for w, v in zip(chunk, emb):
            mapped[w] = v

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(mapped, open(cache_p, 'wb'))
    print('Wrote MPNet text embedding cache to', cache_p)
    return mapped


def load_image_features(num_example):
    candidates = [Path('utils') / f'img_features_{num_example}.pkl', Path('utils') / f'img_features_8.pkl']
    for c in candidates:
        if c.exists():
            return pickle.load(open(c, 'rb'))
    raise FileNotFoundError('No precomputed image features found at expected locations: ' + ','.join(map(str, candidates)))


def compute_topk_predictions(img_features, text_emb_map, words, top_k=5):
    X = np.array(img_features)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X.reshape(X.shape[0], -1)
    W = np.stack([text_emb_map[w] for w in words], axis=0)
    img_norms = np.linalg.norm(X, axis=1, keepdims=True)
    img_norms[img_norms == 0] = 1.0
    Xn = X / img_norms
    sims = Xn @ W.T
    preds = []
    for i in range(sims.shape[0]):
        top_idx = np.argsort(sims[i])[::-1][:top_k]
        preds.append([words[j] for j in top_idx])
    return preds


def compute_patch_embeddings(image_path, model, preprocess, device='cpu', patch_size=224, stride=64, batch=64):
    """Compute CLIP image embeddings for sliding-window patches over an image.

    Returns: (Hsteps, Wsteps, D) numpy array of L2-normalized vectors.
    """
    if Image is None:
        raise RuntimeError('PIL not available; cannot compute spatial heatmaps')
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    # ensure last patch covers right/bottom edge
    if xs[-1] + patch_size < w:
        xs.append(w - patch_size)
    if ys[-1] + patch_size < h:
        ys.append(h - patch_size)

    patches = []
    coords = []
    for y in ys:
        for x in xs:
            box = (x, y, x + patch_size, y + patch_size)
            crop = img.crop(box)
            patches.append(preprocess(crop))
            coords.append((x, y))

    if len(patches) == 0:
        return np.zeros((0, 0, model.visual.output_dim if hasattr(model.visual, 'output_dim') else 512))

    emb_list = []
    for i in range(0, len(patches), batch):
        chunk = torch.stack(patches[i:i+batch]).to(device)
        with torch.no_grad():
            e = model.encode_image(chunk).cpu().numpy()
        emb_list.append(e)
    emb = np.vstack(emb_list)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-10, None)

    D = emb.shape[1]
    Hs = len(ys)
    Ws = len(xs)
    emb_grid = emb.reshape(Hs, Ws, D)
    return emb_grid, xs, ys


def generate_spatial_heatmaps_for_neurons(image_paths, neuron_vectors, out_dir='reports/heatmaps_spatial',
                                          device='cpu', patch_size=224, stride=64, batch=32,
                                          text_words=None, text_emb_map=None, mpnet_emb_map=None, mpnet_alpha=0.5):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    if clip is None or torch is None:
        raise RuntimeError('CLIP not available')

    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()

    # normalize neuron vectors
    V = np.array(neuron_vectors)
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Vn = V / nrm

    for img_path in tqdm(image_paths, desc='Images for spatial heatmaps'):
        try:
            emb_grid, xs, ys = compute_patch_embeddings(img_path, model, preprocess, device=device, patch_size=patch_size, stride=stride, batch=batch)
        except Exception as e:
            print('Skipping', img_path, 'due to', e)
            continue
        Hs, Ws, D = emb_grid.shape
        # compute similarity between each neuron and each patch: (num_neurons, Hs, Ws)
        sims = np.tensordot(Vn, emb_grid, axes=([1], [2]))  # shape (num_neurons, Hs, Ws)

        imgname = Path(img_path).stem
        for ni in range(sims.shape[0]):
            heat = sims[ni]  # Hs x Ws
            np.save(outp / f'neuron_{ni:04d}_image_{imgname}_clip.npy', heat)
            if plt is not None:
                plt.figure(figsize=(6, 6 * (heat.shape[0] / heat.shape[1]) if heat.shape[1] else 6))
                plt.imshow(heat, cmap='viridis')
                plt.colorbar()
                plt.title(f'Neuron {ni} spatial (CLIP) on {imgname}')
                plt.tight_layout()
                plt.savefig(outp / f'neuron_{ni:04d}_image_{imgname}_clip.png', dpi=150)
                plt.close()

            # If MPNet re-ranking is requested we create an MPNet-prototype and a combined map
            if mpnet_emb_map is not None and text_words is not None and text_emb_map is not None:
                # For this neuron, pick top-k words by CLIP (global across vocabulary) to build a prototype
                W = np.stack([text_emb_map[w] for w in text_words], axis=0)
                neuron_clip_scores = np.dot(Vn[ni:ni+1], W.T).reshape(-1)
                top_idx = np.argsort(neuron_clip_scores)[::-1][:min(20, len(text_words))]
                top_words = [text_words[i] for i in top_idx]
                # build MPNet prototype: average MPNet embeddings of top words
                proto = np.mean([mpnet_emb_map[w] for w in top_words], axis=0)
                proto = proto / np.linalg.norm(proto)
                # compute MPNet score for each top word and create a re-ranking weight per word
                # then combine by mapping each patch's best CLIP word to MPNet re-rank score
                # Simpler approach: for each patch, find its top CLIP word (from global vocab) and assign mpnet score
                # Compute per-patch top word indices
                # First, compute patch embeddings flattened
                flat_patches = emb_grid.reshape(-1, D)
                # patch vs vocab CLIP scores
                patch_scores_vocab = flat_patches @ W.T  # (num_patches, V)
                patch_top_idx = np.argmax(patch_scores_vocab, axis=1)
                # mpnet scores for vocab words
                mp_vocab = np.stack([mpnet_emb_map[w] for w in text_words], axis=0)
                # cosine between proto and each vocab word
                mp_sim_vocab = mp_vocab @ proto
                # per-patch mpnet score (based on its top CLIP word)
                patch_mp_scores = mp_sim_vocab[patch_top_idx]
                # reshape back to Hs x Ws
                mp_map = patch_mp_scores.reshape(Hs, Ws)
                np.save(outp / f'neuron_{ni:04d}_image_{imgname}_mpnet.npy', mp_map)
                if plt is not None:
                    plt.figure(figsize=(6, 6 * (mp_map.shape[0] / mp_map.shape[1]) if mp_map.shape[1] else 6))
                    plt.imshow(mp_map, cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Neuron {ni} spatial (MPNet-proto) on {imgname}')
                    plt.tight_layout()
                    plt.savefig(outp / f'neuron_{ni:04d}_image_{imgname}_mpnet.png', dpi=150)
                    plt.close()

                # combined map: normalize both maps to [0,1] and weighted sum
                clip_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
                mp_norm = (mp_map - mp_map.min()) / (mp_map.max() - mp_map.min() + 1e-12)
                combined = ( (mpnet_alpha * clip_norm) + ((1.0 - mpnet_alpha) * mp_norm) )
                np.save(outp / f'neuron_{ni:04d}_image_{imgname}_combined.npy', combined)
                if plt is not None:
                    plt.figure(figsize=(6, 6 * (combined.shape[0] / combined.shape[1]) if combined.shape[1] else 6))
                    plt.imshow(combined, cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Neuron {ni} spatial (combined) on {imgname}')
                    plt.tight_layout()
                    plt.savefig(outp / f'neuron_{ni:04d}_image_{imgname}_combined.png', dpi=150)
                    plt.close()

    return outp


def generate_topword_heatmaps(img_features, text_emb_map, words, out_dir='reports/heatmaps', top_words=50):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    X = np.array(img_features)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X.reshape(X.shape[0], -1)
    W = np.stack([text_emb_map[w] for w in words], axis=0)
    img_norms = np.linalg.norm(X, axis=1, keepdims=True)
    img_norms[img_norms == 0] = 1.0
    Xn = X / img_norms
    sims = Xn @ W.T

    for i in range(sims.shape[0]):
        row = sims[i]
        top_idx = np.argsort(row)[::-1][:top_words]
        top_words_list = [words[j] for j in top_idx]
        top_vals = row[top_idx]
        np.save(outp / f'neuron_{i:04d}_sims.npy', top_vals)
        if plt is not None:
            plt.figure(figsize=(6, max(2, top_words / 6)))
            arr = top_vals.reshape(1, -1)
            plt.imshow(arr, aspect='auto', cmap='viridis')
            plt.yticks([])
            plt.xticks(range(len(top_words_list)), top_words_list, rotation=90, fontsize=6)
            plt.title(f'Neuron {i} top-{top_words} sims')
            plt.tight_layout()
            plt.savefig(outp / f'neuron_{i:04d}_sims.png', dpi=150)
            plt.close()
        else:
            with open(outp / f'neuron_{i:04d}_sims.csv', 'w', encoding='utf-8') as f:
                for w, v in zip(top_words_list, top_vals):
                    f.write(f"{w}\t{float(v)}\n")
    return outp


def main():
    args = parse_args()
    device = args.device
    # Step 1: load words
    words = load_tiny_words(args.num_words)
    words = words[:args.num_words]

    # Step 2: compute or load CLIP text embeddings
    text_emb_map = compute_clip_text_embeddings(words, args.template, args.cache, device=device)

    # Step 2b: optionally load MPNet text embeddings
    mpnet_emb_map = None
    if args.mpnet:
        mpnet_emb_map = load_mpnet_text_embeddings(words, args.mpnet_cache, device=device)

    # Step 3: load image/neuron features (precomputed)
    img_features = load_image_features(args.num_example)

    # Step 4: select top-k per image/neuron
    preds = compute_topk_predictions(img_features[:args.num_neurons], text_emb_map, words, top_k=args.top_k)

    # Heatmap generation (legacy top-word)
    if getattr(args, 'heatmaps', False) and not args.spatial:
        print('Generating top-word heatmaps...')
        hm_out = generate_topword_heatmaps(img_features[:args.num_neurons], text_emb_map, words, out_dir='reports/heatmaps', top_words=args.heatmap_top)
        print('Top-word heatmaps written to', hm_out)

    # Spatial heatmaps across example images (sliding-window) if requested
    if getattr(args, 'heatmaps', False) and args.spatial:
        print('Generating spatial (patch-level) heatmaps...')
        # gather a small set of example images from the repo `images/` folder
        images_dir = Path('images') / 'example_val_fc'
        if not images_dir.exists():
            print('Images directory not found at', images_dir, '; cannot generate spatial heatmaps')
        else:
            # collect a handful of images (first 50 by default)
            img_paths = [str(p) for p in sorted(images_dir.rglob('*.jpg'))][:50]
            hm_out = generate_spatial_heatmaps_for_neurons(img_paths, img_features[:args.num_neurons], out_dir='reports/heatmaps_spatial',
                                                          device=device, patch_size=224, stride=64, batch=32,
                                                          text_words=words, text_emb_map=text_emb_map,
                                                          mpnet_emb_map=mpnet_emb_map, mpnet_alpha=args.mpnet_alpha)
            print('Spatial heatmaps written to', hm_out)

    # Save predictions (list-of-lists)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(outp, 'wb'))
    print('Wrote predictions to', outp)

    # Write small GT file (first N lines) if full GT exists
    gt_full = Path('reports') / 'gt_labels.txt'
    small_gt = Path('reports') / 'gt_labels_small_reimpl.txt'
    if gt_full.exists():
        lines = [l.strip() for l in open(gt_full, 'r', encoding='utf-8').read().splitlines() if l.strip()]
        small = lines[:args.num_neurons]
        small_gt.write_text('\n'.join(small), encoding='utf-8')
        print('Wrote small GT to', small_gt)
    else:
        print('Full GT not found; skipping GT write')

    # Optional evaluation
    if args.eval:
        cmd = [sys.executable, 'compute_score.py', '--gt', str(small_gt.resolve()), '--target_pkl', str(outp.resolve()), '--device', args.device, '--metric', 'acc,sim,pr', '--save_dir', 'results']
        print('Running evaluation:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Write a tiny report summarizing first few preds for quick inspection
    report = {
        'num_neurons': args.num_neurons,
        'num_words': args.num_words,
        'top_k': args.top_k,
        'sample_preds': preds[:5]
    }
    Path('reports').mkdir(parents=True, exist_ok=True)
    Path('reports/reimpl_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote reports/reimpl_summary.json')


if __name__ == '__main__':
    main()
