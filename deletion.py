from __future__ import annotations

from model import build_model
from data import embedding_collate
from main import amp_autocast, build_loader, collect_binary_scores, collect_threshold_report, load_config, move_batch

import math
from collections import defaultdict
import numpy as np
import torch

def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}

def delete_ranked_positions(batch: dict, ranking: torch.Tensor, ratio: float, embedding_key: str, mask_key: str, use_ceil: bool) -> tuple[dict, int]:
    masked_batch = clone_batch(batch)
    raw_count = ranking.numel() * ratio
    count = int(np.ceil(raw_count) if use_ceil else np.floor(raw_count))
    indices = ranking[:count]
    if count:
        masked_batch[embedding_key][0, indices] = 0.0
        masked_batch[mask_key][0, indices] = True
    return (masked_batch, count)

def validate_protocol_args(ratios: list[float], random_repeats: int, max_pos: int, max_neg: int) -> None:
    if not ratios:
        raise ValueError('--ratios must contain at least one value.')
    if any((not math.isfinite(value) or value < 0.0 or value > 1.0 for value in ratios)):
        raise ValueError('Every deletion ratio must be finite and within [0, 1].')
    if ratios[0] != 0.0:
        raise ValueError('--ratios must start at 0 so the deletion curve has a baseline.')
    if any((right <= left for left, right in zip(ratios, ratios[1:]))):
        raise ValueError('--ratios must be strictly increasing without duplicates.')
    if random_repeats < 1:
        raise ValueError('--random_repeats must be at least 1.')
    if max_pos < 0 or max_neg < 0 or max_pos + max_neg == 0:
        raise ValueError('--max_pos and --max_neg must be non-negative and not both zero.')

def summarize_rows(rows: list[dict], ratios: list[float]) -> tuple[list[dict], dict]:
    per_sample: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (row['strategy'], row['sample_id'], row['label_group'], float(row['ratio']))
        per_sample[key].append(float(row['margin_change']))
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for (strategy, _sample_id, label_group, ratio), values in per_sample.items():
        grouped[strategy, label_group, ratio].append(float(np.mean(values)))
    summary_rows = []
    audc: dict[str, dict[str, float]] = defaultdict(dict)
    for label_group in ('pos', 'neg'):
        for strategy in ('selective', 'random'):
            means = []
            for ratio in ratios:
                values = np.asarray(grouped.get((strategy, label_group, float(ratio)), []), dtype=np.float64)
                mean = float(values.mean()) if values.size else float('nan')
                std = float(values.std(ddof=0)) if values.size else float('nan')
                means.append(mean)
                summary_rows.append({'label_group': label_group, 'strategy': strategy, 'ratio': ratio, 'n_samples': int(values.size), 'margin_change_mean': mean, 'margin_change_std': std})
            if np.all(np.isfinite(means)):
                trapezoid = getattr(np, 'trapezoid', None)
                if trapezoid is None:
                    trapezoid = np.trapz
                audc[label_group][strategy] = float(trapezoid(means, ratios))
            else:
                audc[label_group][strategy] = float('nan')
    return (summary_rows, {key: dict(value) for key, value in audc.items()})
import matplotlib as mpl
BIB_FIGURE_WIDTH = 7.2
BIB_FONT_SIZE = 8.0
BIB_SMALL_FONT_SIZE = 7.5

def apply_bib_figure_style() -> None:
    """Apply a consistent, print-sized style to manuscript figures."""
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'], 'font.size': BIB_FONT_SIZE, 'axes.titlesize': BIB_FONT_SIZE, 'axes.labelsize': BIB_FONT_SIZE, 'xtick.labelsize': BIB_SMALL_FONT_SIZE, 'ytick.labelsize': BIB_SMALL_FONT_SIZE, 'legend.fontsize': BIB_SMALL_FONT_SIZE, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def run_model(model, batch, prior_scale):
    return model(batch['drug_embedding'], batch['protein_embedding'], batch['drug_mask'], batch['protein_mask'], prior_scale=prior_scale, graph_node_features=batch.get('graph_node_features'), graph_adjacency=batch.get('graph_adjacency'), graph_edge_features=batch.get('graph_edge_features'), graph_node_mask=batch.get('graph_node_mask'), protein_sequence=batch.get('protein_sequence'), protein_sequence_mask=batch.get('protein_sequence_mask'), raw_smiles_tokens=batch.get('raw_smiles_tokens'), raw_smiles_mask=batch.get('raw_smiles_mask'), raw_protein_tokens=batch.get('raw_protein_tokens'), raw_protein_mask=batch.get('raw_protein_mask'))

def ranked_indices(weights: torch.Tensor, mask: torch.Tensor, strategy: str, generator: torch.Generator | None=None) -> torch.Tensor:
    valid = torch.nonzero(~mask, as_tuple=False).flatten()
    if strategy == 'random':
        order = torch.randperm(valid.numel(), generator=generator, device='cpu').to(valid.device)
        return valid[order]
    valid_weights = weights[valid]
    order = torch.argsort(valid_weights, descending=True)
    return valid[order]

@torch.no_grad()
def predict_with_features(model, batch, cfg, device):
    use_amp = cfg['train']['precision'] in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg['train']['precision'] == 'bf16' else torch.float16
    prior_scale = cfg['model'].get('prior_modulation_scale', None)
    with amp_autocast(device.type, enabled=use_amp, dtype=amp_dtype):
        logits = run_model(model, batch, prior_scale=prior_scale)
    protein_weights = getattr(model.protein_pool, 'last_weights', None)
    drug_weights = getattr(model.drug_pool, 'last_weights', None)
    if protein_weights is None:
        raise RuntimeError('protein_pool.last_weights is missing. Copy the patched pooling.py first.')
    if drug_weights is None:
        raise RuntimeError('drug_pool.last_weights is missing. Copy the patched pooling.py first.')
    return (logits.float(), {'protein_pool_weights': protein_weights, 'drug_pool_weights': drug_weights})

@torch.no_grad()
def predict_logits(model, batch, cfg, device):
    use_amp = cfg['train']['precision'] in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg['train']['precision'] == 'bf16' else torch.float16
    prior_scale = cfg['model'].get('prior_modulation_scale', None)
    with amp_autocast(device.type, enabled=use_amp, dtype=amp_dtype):
        logits = run_model(model, batch, prior_scale=prior_scale)
    return logits.float()

def save_plot(summary_rows: list[dict], audc: dict, output_path: Path, side: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for axis, label_group, title in zip(axes, ('pos', 'neg'), ('Positive samples', 'Negative samples')):
        for strategy, color in (('selective', '#C44E52'), ('random', '#4C72B0')):
            selected = [row for row in summary_rows if row['label_group'] == label_group and row['strategy'] == strategy]
            x = np.asarray([row['ratio'] for row in selected], dtype=float)
            y = np.asarray([row['margin_change_mean'] for row in selected], dtype=float)
            std = np.asarray([row['margin_change_std'] for row in selected], dtype=float)
            axis.plot(x, y, marker='o', color=color, label=f'{strategy} (AUDC={audc[label_group][strategy]:.4f})')
            axis.fill_between(x, y - std, y + std, color=color, alpha=0.16)
        axis.axhline(0.0, color='#777777', linewidth=0.8)
        axis.set_title(title)
        unit = 'protein residues' if side == 'protein' else 'drug tokens'
        axis.set_xlabel(f'Fraction of {unit} masked')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('Predicted-class margin decrease')
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def deletion_cli() -> None:
    parser = argparse.ArgumentParser(description='Dataset-level deletion faithfulness experiment.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--side', '--modality', dest='side', default='protein', choices=['protein', 'drug'])
    parser.add_argument('--correct_only', action='store_true', help='Only analyze samples correctly classified by the original model.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_pos', type=int, default=512)
    parser.add_argument('--max_neg', type=int, default=512)
    parser.add_argument('--sample_ids', nargs='+', default=None, help='Only analyze exact drug_id__target_id pairs, for example DB00440__P00374.')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--random_repeats', type=int, default=20, help='Number of independent random-deletion rankings per sample (default: 20).')
    parser.add_argument('--skip_plot', action='store_true', help='Write CSV and JSON results without importing matplotlib.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    validate_protocol_args(args.ratios, args.random_repeats, args.max_pos, args.max_neg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    runtime_cfg = load_config(args.config)
    cfg = checkpoint.get('config') if isinstance(checkpoint, dict) and 'config' in checkpoint else None
    if cfg is None:
        cfg = runtime_cfg
    else:
        cfg['data'] = runtime_cfg['data']
        cfg['embedding'] = runtime_cfg['embedding']
    cfg['train']['batch_size'] = 1
    cfg['train']['progress_bar'] = False
    model = build_model(cfg, device, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    if cfg['train'].get('tune_threshold', True):
        valid_loader = build_loader(cfg, 'valid', shuffle=False)
        y_true, y_score = collect_binary_scores(model, valid_loader, device, cfg)
        threshold, threshold_report = collect_threshold_report(y_true, y_score, selected_metric=cfg['train'].get('threshold_metric', 'mcc'))
    else:
        threshold = float(cfg['train'].get('threshold', 0.5))
        threshold_report = {'threshold': threshold}
    threshold = float(np.clip(threshold, 1e-06, 1.0 - 1e-06))
    threshold_logit = math.log(threshold / (1.0 - threshold))
    if args.side == 'protein':
        embedding_key = 'protein_embedding'
        mask_key = 'protein_mask'
        weight_key = 'protein_pool_weights'
        use_ceil = False
    else:
        embedding_key = 'drug_embedding'
        mask_key = 'drug_mask'
        weight_key = 'drug_pool_weights'
        use_ceil = True
    data_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(build_loader(cfg, args.split, shuffle=False).dataset, batch_size=1, shuffle=True, generator=data_generator, num_workers=0, collate_fn=embedding_collate)
    selected = {'pos': 0, 'neg': 0}
    requested_sample_ids = set(args.sample_ids or [])
    found_sample_ids = set()
    rows = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        if requested_sample_ids and found_sample_ids == requested_sample_ids:
            break
        if not requested_sample_ids and selected['pos'] >= args.max_pos and (selected['neg'] >= args.max_neg):
            break
        sample_id = f"{batch['drug_ids'][0]}__{batch['target_ids'][0]}"
        if requested_sample_ids and sample_id not in requested_sample_ids:
            continue
        batch = move_batch(batch, device)
        logits, features = predict_with_features(model, batch, cfg, device)
        original_logit = float(logits.item())
        probability = float(torch.sigmoid(logits).item())
        label = int(batch['label'].item())
        original_pred = int(probability >= threshold)
        if args.correct_only and original_pred != label:
            continue
        direction = 1.0 if original_pred == 1 else -1.0
        original_margin = direction * (original_logit - threshold_logit)
        label_group = 'pos' if label == 1 else 'neg'
        if selected[label_group] >= (args.max_pos if label_group == 'pos' else args.max_neg):
            continue
        selected[label_group] += 1
        found_sample_ids.add(sample_id)
        weights = features[weight_key][0].detach().clone()
        side_mask = batch[mask_key][0]
        selective_ranking = ranked_indices(weights, side_mask, 'selective')
        for ratio in args.ratios:
            masked_batch, masked_count = delete_ranked_positions(batch, selective_ranking, ratio, embedding_key, mask_key, use_ceil)
            masked_logit = float(predict_logits(model, masked_batch, cfg, device).item())
            masked_probability = float(torch.sigmoid(torch.tensor(masked_logit)).item())
            masked_pred = int(masked_probability >= threshold)
            masked_margin = direction * (masked_logit - threshold_logit)
            rows.append({'side': args.side, 'sample_id': sample_id, 'drug_id': batch['drug_ids'][0], 'target_id': batch['target_ids'][0], 'label': label, 'label_group': label_group, 'original_pred': original_pred, 'original_logit': original_logit, 'probability': probability, 'threshold': threshold, 'threshold_logit': threshold_logit, 'original_margin': original_margin, 'strategy': 'selective', 'ratio': ratio, 'repeat': 0, 'side_length': int(selective_ranking.numel()), 'masked_count': masked_count, 'masked_logit': masked_logit, 'masked_probability': masked_probability, 'masked_pred': masked_pred, 'masked_margin': masked_margin, 'margin_change': original_margin - masked_margin, 'prediction_flip': int(masked_pred != original_pred)})
        for repeat in range(args.random_repeats):
            random_generator = torch.Generator().manual_seed(args.seed + selected['pos'] * 100003 + selected['neg'] * 1009 + repeat)
            random_ranking = ranked_indices(weights, side_mask, 'random', generator=random_generator)
            for ratio in args.ratios:
                masked_batch, masked_count = delete_ranked_positions(batch, random_ranking, ratio, embedding_key, mask_key, use_ceil)
                masked_logit = float(predict_logits(model, masked_batch, cfg, device).item())
                masked_probability = float(torch.sigmoid(torch.tensor(masked_logit)).item())
                masked_pred = int(masked_probability >= threshold)
                masked_margin = direction * (masked_logit - threshold_logit)
                rows.append({'side': args.side, 'sample_id': sample_id, 'drug_id': batch['drug_ids'][0], 'target_id': batch['target_ids'][0], 'label': label, 'label_group': label_group, 'original_pred': original_pred, 'original_logit': original_logit, 'probability': probability, 'threshold': threshold, 'threshold_logit': threshold_logit, 'original_margin': original_margin, 'strategy': 'random', 'ratio': ratio, 'repeat': repeat, 'side_length': int(random_ranking.numel()), 'masked_count': masked_count, 'masked_logit': masked_logit, 'masked_probability': masked_probability, 'masked_pred': masked_pred, 'masked_margin': masked_margin, 'margin_change': original_margin - masked_margin, 'prediction_flip': int(masked_pred != original_pred)})
    if not rows:
        raise RuntimeError('No samples were selected from the requested split.')
    missing_sample_ids = requested_sample_ids - found_sample_ids
    if missing_sample_ids:
        raise RuntimeError('Requested samples were not selected: ' + ', '.join(sorted(missing_sample_ids)))
    detail_path = out_dir / 'deletion_detail.csv'
    with open(detail_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows, audc = summarize_rows(rows, args.ratios)
    summary_path = out_dir / 'deletion_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {'selected': selected, 'side': args.side, 'correct_only': args.correct_only, 'protocol': f'3DICE-style {args.side} deletion', 'score': 'decrease in predicted-class margin', 'threshold_report': threshold_report, 'threshold_logit': threshold_logit, 'ratios': args.ratios, 'random_repeats': args.random_repeats, 'audc': audc, 'detail_csv': str(detail_path), 'summary_csv': str(summary_path)}
    if not args.skip_plot:
        plot_path = out_dir / 'deletion_curves.png'
        save_plot(summary_rows, audc, plot_path, args.side)
        report['plot'] = str(plot_path)
    with open(out_dir / 'deletion_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
COLORS = {'selective': '#1f77b4', 'random': '#ff7f0e'}

def load_summary(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    required = {'label_group', 'strategy', 'ratio', 'margin_change_mean', 'margin_change_std'}
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0] if rows else {}))
        raise ValueError(f'{path} is missing required columns: {missing}')
    return rows

def curve(rows: list[dict], label_group: str, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted((row for row in rows if row['label_group'] == label_group and row['strategy'] == strategy), key=lambda row: float(row['ratio']))
    x = np.asarray([float(row['ratio']) for row in selected], dtype=float)
    mean = np.asarray([float(row['margin_change_mean']) for row in selected], dtype=float)
    std = np.asarray([float(row['margin_change_std']) for row in selected], dtype=float)
    return (x, mean, std)

def audc(x: np.ndarray, y: np.ndarray) -> float:
    trapezoid = getattr(np, 'trapezoid', None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x))

def draw_panel(ax, rows: list[dict], modality: str, label_group: str) -> None:
    values = {}
    for strategy in ('selective', 'random'):
        x, mean, std = curve(rows, label_group, strategy)
        if x.size == 0:
            raise ValueError(f'No {strategy}/{label_group} rows found for {modality}')
        values[strategy] = audc(x, mean)
        ax.plot(x, mean, color=COLORS[strategy], linewidth=1.5, label=strategy)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[strategy], alpha=0.2, linewidth=0)
    group = 'POS' if label_group == 'pos' else 'NEG'
    ax.set_title(f"{modality} deletion ({group})\nAUDC selective={values['selective']:.4f}, random={values['random']:.4f}", fontsize=BIB_FONT_SIZE, pad=5)
    ax.set_xlabel('fraction masked')
    ax.set_ylabel('Δmargin')
    ax.axhline(0.0, color='#555555', linewidth=0.7, alpha=0.8)
    ax.grid(True, color='#d9d9d9', linewidth=0.55, alpha=0.55)
    ax.legend(loc='upper left', frameon=False, fontsize=BIB_SMALL_FONT_SIZE)
    ax.tick_params(labelsize=BIB_SMALL_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#444444')

def plot_cli() -> None:
    parser = argparse.ArgumentParser(description='Create a 3DICE-style 2x2 protein/drug deletion figure.')
    parser.add_argument('--protein_summary', required=True)
    parser.add_argument('--drug_summary', required=True)
    parser.add_argument('--output', required=True, help='Output path without extension.')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    protein_rows = load_summary(args.protein_summary)
    drug_rows = load_summary(args.drug_summary)
    apply_bib_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(BIB_FIGURE_WIDTH, 5.0))
    draw_panel(axes[0, 0], protein_rows, 'Protein', 'pos')
    draw_panel(axes[0, 1], protein_rows, 'Protein', 'neg')
    draw_panel(axes[1, 0], drug_rows, 'Drug-token', 'pos')
    draw_panel(axes[1, 1], drug_rows, 'Drug-token', 'neg')
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {output.with_suffix('.png')}")
    print(f"saved: {output.with_suffix('.pdf')}")
    print(f"saved: {output.with_suffix('.svg')}")
    print(f"saved: {output.with_suffix('.tiff')}")

if __name__ == "__main__":
    import sys
    commands = {"run": deletion_cli, "plot": plot_cli}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python deletion.py {run|plot} [arguments]")
    command = commands[sys.argv.pop(1)]
    command()
