#!/usr/bin/env python3
"""Compute CLIP / MPNet semantic-match at threshold 0.5 for Traditional and WWW predictions.

Usage (from repo root):
  .venv/bin/python scripts/compute_semantic_vs_paper_0.5.py \
      --gt reports/gt_labels.txt \
      --trad reports/ours_predictions.pkl \
      --www utils/www_tinyk_tem_adp_95_preds_103.pkl \
      --threshold 0.5 \
      --out reports/semantic_vs_paper_0.5.json

This reproduces the quick semantic comparison run that was used interactively in the session
and writes a JSON summary with counts and accuracies.
"""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', required=True)
    p.add_argument('--trad', required=True)
    p.add_argument('--www', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--out', default='reports/semantic_vs_paper_0.5.json')
    return p.parse_args()


def clip_text_embs(model, texts, device='cpu', batch=64):
    embs = {}
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        toks = clip.tokenize(chunk).to(device)
        with torch.no_grad():
            e = model.encode_text(toks).cpu()
        e = e / e.norm(dim=1, keepdim=True)
        for s, v in zip(chunk, e):
            embs[s] = v
    return embs


def main():
    args = parse_args()
    gt_lines = [l.strip() for l in open(args.gt, 'r', encoding='utf-8').read().splitlines() if l.strip()]
    trad_preds = pickle.load(open(args.trad, 'rb'))
    www_preds = pickle.load(open(args.www, 'rb'))
    N = len(gt_lines)
    trad_preds = trad_preds[:N]
    www_preds = www_preds[:N]

    device = 'cpu'
    clip_model, _ = clip.load('ViT-B/16', device=device)
    mpnet = SentenceTransformer('all-mpnet-base-v2')

    all_gt_alternatives = [[g.strip() for g in gt_lines[i].split(',')] for i in range(N)]
    flat_gt = list({s for subs in all_gt_alternatives for s in subs if s})

    trad_strs = list({p for lst in trad_preds for p in lst if isinstance(p, str) and p})
    www_strs = list({p for lst in www_preds for p in lst if isinstance(p, str) and p})
    all_pred_strs = list(set(trad_strs) | set(www_strs))

    gt_clip = clip_text_embs(clip_model, flat_gt, device=device)
    pred_clip = clip_text_embs(clip_model, all_pred_strs, device=device)

    gt_mp_list = mpnet.encode(flat_gt, convert_to_tensor=False)
    gt_mp = {s: v for s, v in zip(flat_gt, gt_mp_list)}
    pred_mp_list = mpnet.encode(all_pred_strs, convert_to_tensor=False)
    pred_mp = {s: v for s, v in zip(all_pred_strs, pred_mp_list)}

    def compute_scores(preds_list, model_name='CLIP'):
        scores = []
        for i in range(N):
            best = -1.0
            for p in preds_list[i]:
                if not isinstance(p, str) or p.strip() == '':
                    continue
                p = p.strip()
                for g in all_gt_alternatives[i]:
                    g = g.strip()
                    if g == '':
                        continue
                    if model_name == 'CLIP':
                        if p in pred_clip and g in gt_clip:
                            sim = float((pred_clip[p] @ gt_clip[g]).sum())
                        else:
                            sim = -1.0
                    else:
                        pv = pred_mp.get(p)
                        gv = gt_mp.get(g)
                        if pv is None or gv is None:
                            sim = -1.0
                        else:
                            denom = (np.linalg.norm(pv) * np.linalg.norm(gv))
                            sim = float(np.dot(pv, gv) / denom) if denom > 0 else -1.0
                    if sim > best:
                        best = sim
            scores.append(best)
        return np.array(scores)

    clip_scores_trad = compute_scores(trad_preds, 'CLIP')
    clip_scores_www = compute_scores(www_preds, 'CLIP')
    mp_scores_trad = compute_scores(trad_preds, 'MPNet')
    mp_scores_www = compute_scores(www_preds, 'MPNet')

    th = args.threshold
    results = {
        'threshold': th,
        'N': N,
        'Traditional': {
            'CLIP_count': int((clip_scores_trad >= th).sum()),
            'CLIP_acc': float((clip_scores_trad >= th).mean()),
            'MPNet_count': int((mp_scores_trad >= th).sum()),
            'MPNet_acc': float((mp_scores_trad >= th).mean())
        },
        'WWW': {
            'CLIP_count': int((clip_scores_www >= th).sum()),
            'CLIP_acc': float((clip_scores_www >= th).mean()),
            'MPNet_count': int((mp_scores_www >= th).sum()),
            'MPNet_acc': float((mp_scores_www >= th).mean())
        }
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {outp}')


if __name__ == '__main__':
    main()
