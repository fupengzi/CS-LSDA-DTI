from __future__ import annotations

from model import build_model
from data import EmbeddingDataset, embedding_collate, convert_dataset

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.special import expit
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from torch.amp import GradScaler, autocast
    _AMP_REQUIRES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_REQUIRES_DEVICE_TYPE = False

def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def make_grad_scaler(enabled: bool) -> GradScaler:
    if _AMP_REQUIRES_DEVICE_TYPE:
        return GradScaler('cuda', enabled=enabled)
    return GradScaler(enabled=enabled)

def amp_autocast(device_type: str, enabled: bool, dtype: torch.dtype):
    if _AMP_REQUIRES_DEVICE_TYPE:
        return autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return autocast(enabled=enabled, dtype=dtype)

def atomic_torch_save(payload: object, path: Path) -> None:
    """Keep the previous checkpoint intact if a new save is interrupted."""
    temp = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()

def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg['data']
    emb_cfg = cfg['embedding']
    split_path = Path(data_cfg['split_dir']) / f'{split}.csv'
    if split == 'valid' and (not split_path.exists()):
        split_path = Path(data_cfg['split_dir']) / 'validation.csv'
    dataset = EmbeddingDataset(csv_path=split_path, drug_embedding_dir=emb_cfg['drug_embedding_dir'], protein_embedding_dir=emb_cfg['protein_embedding_dir'], max_drug_len=data_cfg['max_drug_len'], max_protein_len=data_cfg['max_protein_len'], use_multiview_residual=cfg['model'].get('use_multiview_residual', False), max_graph_atoms=cfg['model'].get('max_graph_atoms', 128), max_raw_protein_len=cfg['model'].get('max_raw_protein_len', data_cfg['max_protein_len']), drug_encoder_type=cfg['model'].get('drug_encoder_type', 'plm'), protein_encoder_type=cfg['model'].get('protein_encoder_type', 'plm'))
    return DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=shuffle, num_workers=cfg['train'].get('num_workers', 0), pin_memory=torch.cuda.is_available(), collate_fn=embedding_collate)

def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def forward_model(model: torch.nn.Module, batch: dict, prior_scale: float | None=None, return_features: bool=False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return model(batch['drug_embedding'], batch['protein_embedding'], batch['drug_mask'], batch['protein_mask'], prior_scale=prior_scale, graph_node_features=batch.get('graph_node_features'), graph_adjacency=batch.get('graph_adjacency'), graph_edge_features=batch.get('graph_edge_features'), graph_node_mask=batch.get('graph_node_mask'), protein_sequence=batch.get('protein_sequence'), protein_sequence_mask=batch.get('protein_sequence_mask'), raw_smiles_tokens=batch.get('raw_smiles_tokens'), raw_smiles_mask=batch.get('raw_smiles_mask'), raw_protein_tokens=batch.get('raw_protein_tokens'), raw_protein_mask=batch.get('raw_protein_mask'), return_features=return_features)

def compute_loss(pred: torch.Tensor, label: torch.Tensor, cfg: dict) -> torch.Tensor:
    if cfg['train']['task'] in {'binary', 'binary_classification', 'classification'}:
        positive_weight = cfg.get('loss', {}).get('positive_weight', None)
        pos_weight = None
        if positive_weight is not None:
            pos_weight = torch.as_tensor(float(positive_weight), device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight)
    mse = F.mse_loss(pred, label)
    mae = F.l1_loss(pred, label)
    return cfg['loss']['mse_weight'] * mse + cfg['loss']['mae_weight'] * mae

def pair_contrastive_loss(drug_pool: torch.Tensor, protein_pool: torch.Tensor, label: torch.Tensor, temperature: float) -> torch.Tensor:
    drug = F.normalize(drug_pool, dim=-1)
    protein = F.normalize(protein_pool, dim=-1)
    logits = (drug * protein).sum(dim=-1) / max(float(temperature), 1e-06)
    return F.binary_cross_entropy_with_logits(logits, label)

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    use_amp = cfg['train']['precision'] in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg['train']['precision'] == 'bf16' else torch.float16
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            pred = forward_model(model, batch, prior_scale=cfg['model'].get('prior_modulation_scale', None))
            loss = compute_loss(pred, batch['label'], cfg)
        losses.append(float(loss.item()))
        preds.append(pred.detach().float().cpu().numpy())
        labels.append(batch['label'].detach().float().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    if cfg['train']['task'] in {'binary', 'binary_classification', 'classification'}:
        y_score = expit(y_pred)
        metrics = binary_classification_metrics(y_true, y_score, threshold=cfg['train'].get('threshold', 0.5))
    else:
        metrics = regression_metrics(y_true, y_pred)
    metrics['loss'] = float(np.mean(losses))
    return metrics

@torch.no_grad()
def collect_binary_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    use_amp = cfg['train']['precision'] in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg['train']['precision'] == 'bf16' else torch.float16
    prior_scale = cfg['model'].get('prior_modulation_scale', None)
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            pred = forward_model(model, batch, prior_scale=prior_scale)
        preds.append(pred.detach().float().cpu().numpy())
        labels.append(batch['label'].detach().float().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_score = expit(y_pred)
    y_true = np.concatenate(labels)
    return (y_true, y_score)

def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def collect_threshold_report(y_true: np.ndarray, y_score: np.ndarray, selected_metric: str) -> tuple[float, dict]:
    report: dict[str, float | str] = {'selected_metric': selected_metric}
    selected_threshold = 0.5
    selected_value = -float('inf')
    for metric in ('mcc', 'f1'):
        threshold, value = find_best_threshold(y_true, y_score, metric=metric)
        report[f'best_{metric}_threshold'] = threshold
        report[f'best_{metric}'] = value
        if metric == selected_metric:
            selected_threshold = threshold
            selected_value = value
    if selected_metric not in {'mcc', 'f1'}:
        selected_threshold, selected_value = find_best_threshold(y_true, y_score, metric=selected_metric)
    report['threshold'] = selected_threshold
    report[selected_metric] = selected_value
    return (selected_threshold, report)

def train_cli() -> None:
    parser = argparse.ArgumentParser(description='Train CS-LSDA-DTI on cached embeddings.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output_dir', default='outputs/cs_lsda_dti_biosnap')
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = cfg['model']
    drug_encoder_type = model_cfg.get('drug_encoder_type', 'plm')
    protein_encoder_type = model_cfg.get('protein_encoder_type', 'plm')
    print(f'input encoders: drug={drug_encoder_type}, protein={protein_encoder_type}')
    model = build_model(cfg, device)
    train_loader = build_loader(cfg, 'train', shuffle=True)
    valid_loader = build_loader(cfg, 'valid', shuffle=False)
    test_loader = build_loader(cfg, 'test', shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    total_steps = len(train_loader) * cfg['train']['epochs']
    scheduler = make_scheduler(optimizer, total_steps, cfg['train']['warmup_ratio'])
    scaler = make_grad_scaler(enabled=cfg['train']['precision'] == 'fp16' and device.type == 'cuda')
    use_amp = cfg['train']['precision'] in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg['train']['precision'] == 'bf16' else torch.float16
    monitor_metric = cfg['train'].get('monitor_metric', 'auc' if cfg['train']['task'] in {'binary', 'binary_classification', 'classification'} else 'mse')
    monitor_mode = cfg['train'].get('monitor_mode', 'max' if monitor_metric in {'auc', 'aupr', 'accuracy', 'f1', 'sensitivity', 'specificity', 'mcc'} else 'min')
    best_valid = -float('inf') if monitor_mode == 'max' else float('inf')
    best_epoch = 0
    best_valid_metrics: dict[str, float] | None = None
    best_path = output_dir / 'best_model.pt'
    progress_bar = cfg['train'].get('progress_bar', False)
    for epoch in range(1, cfg['train']['epochs'] + 1):
        model.train()
        running = []
        iterator = tqdm(train_loader, desc=f'epoch {epoch}') if progress_bar else train_loader
        optimizer.zero_grad(set_to_none=True)
        prior_warmup_epochs = cfg['train'].get('prior_warmup_epochs', 0)
        prior_base_scale = model_cfg.get('prior_modulation_scale', 0.5)
        if model_cfg.get('use_pseudo_binding_prior', True) and prior_warmup_epochs > 0:
            prior_scale = prior_base_scale * min(1.0, epoch / prior_warmup_epochs)
        else:
            prior_scale = prior_base_scale
        for batch in iterator:
            batch = move_batch(batch, device)
            with amp_autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                contrastive_weight = float(cfg.get('loss', {}).get('contrastive_weight', 0.0) or 0.0)
                if contrastive_weight > 0:
                    pred, features = forward_model(model, batch, prior_scale=prior_scale, return_features=True)
                else:
                    pred = forward_model(model, batch, prior_scale=prior_scale)
                loss = compute_loss(pred, batch['label'], cfg)
                if contrastive_weight > 0:
                    contrastive = pair_contrastive_loss(features['drug_pool'], features['protein_pool'], batch['label'], temperature=cfg.get('loss', {}).get('contrastive_temperature', 0.2))
                    loss = loss + contrastive_weight * contrastive
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(float(loss.item()))
            if progress_bar:
                iterator.set_postfix(loss=np.mean(running))
        train_loss = float(np.mean(running))
        print(f'epoch {epoch} train {{"loss": {train_loss:.8f}}}', flush=True)
        valid_metrics = evaluate(model, valid_loader, device, cfg)
        print(f'epoch {epoch} valid {json.dumps(valid_metrics, indent=None)}', flush=True)
        current_valid = valid_metrics[monitor_metric]
        improved = current_valid > best_valid if monitor_mode == 'max' else current_valid < best_valid
        if improved:
            best_valid = current_valid
            best_epoch = epoch
            best_valid_metrics = {key: float(value) for key, value in valid_metrics.items()}
            checkpoint_info = {'best_epoch': best_epoch, 'monitor_metric': monitor_metric, 'monitor_mode': monitor_mode, 'best_valid': float(best_valid), 'valid': best_valid_metrics}
            atomic_torch_save({'model': model.state_dict(), 'config': cfg, 'best': checkpoint_info}, best_path)
            print(f'best_update epoch={best_epoch} {monitor_metric}={float(current_valid):.8f} metrics={json.dumps(valid_metrics, ensure_ascii=False)}', flush=True)
    if not best_path.exists():
        raise RuntimeError(f'No best checkpoint was produced. Check whether {monitor_metric} was finite on the validation split.')
    checkpoint = torch.load(best_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    best_checkpoint = checkpoint.get('best', {'best_epoch': best_epoch, 'monitor_metric': monitor_metric, 'monitor_mode': monitor_mode, 'best_valid': float(best_valid), 'valid': best_valid_metrics})
    print(f"using_best_checkpoint epoch={best_checkpoint.get('best_epoch')} {best_checkpoint.get('monitor_metric', monitor_metric)}={float(best_checkpoint.get('best_valid', best_valid)):.8f}", flush=True)
    if cfg['train']['task'] in {'binary', 'binary_classification', 'classification'} and cfg['train'].get('tune_threshold', True):
        y_true, y_score = collect_binary_scores(model, valid_loader, device, cfg)
        threshold_metric = cfg['train'].get('threshold_metric', 'mcc')
        best_threshold, threshold_report = collect_threshold_report(y_true, y_score, selected_metric=threshold_metric)
        cfg['train']['threshold'] = best_threshold
        print(f'best_threshold {json.dumps(threshold_report, ensure_ascii=False)}', flush=True)
    test_metrics = evaluate(model, test_loader, device, cfg)
    metrics_path = output_dir / 'test_metrics.json'
    metrics_temp = metrics_path.with_suffix('.json.tmp')
    with open(metrics_temp, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)
    metrics_temp.replace(metrics_path)
    print(f'test {json.dumps(test_metrics, indent=None)}', flush=True)
import argparse
import json
from pathlib import Path
import torch

def evaluate_cli() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved CS-LSDA-DTI checkpoint.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pt.')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--output_json', default=None)
    args = parser.parse_args()
    runtime_cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    checkpoint_cfg = checkpoint.get('config') if isinstance(checkpoint, dict) else None
    if checkpoint_cfg is None:
        cfg = runtime_cfg
    else:
        cfg = json.loads(json.dumps(checkpoint_cfg))
        cfg['data'] = runtime_cfg['data']
        cfg['embedding'] = runtime_cfg['embedding']
    model = build_model(cfg, device, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    if cfg['train']['task'] in {'binary', 'binary_classification', 'classification'} and cfg['train'].get('tune_threshold', True):
        valid_loader = build_loader(cfg, 'valid', shuffle=False)
        y_true, y_score = collect_binary_scores(model, valid_loader, device, cfg)
        best_threshold, threshold_report = collect_threshold_report(y_true, y_score, selected_metric=cfg['train'].get('threshold_metric', 'mcc'))
        cfg['train']['threshold'] = best_threshold
        print(f'best_threshold {json.dumps(threshold_report, ensure_ascii=False)}')
    loader = build_loader(cfg, args.split, shuffle=False)
    metrics = evaluate(model, loader, device, cfg)
    print(f'{args.split} {json.dumps(metrics, ensure_ascii=False)}')
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import TextIO
DEFAULT_SCENARIOS = ['random', 'e2', 'e3', 'e4']

def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

def parse_json_line(line: str, prefix: str) -> dict | None:
    if not line.startswith(prefix):
        return None
    return json.loads(line[len(prefix):].strip())

def now_text() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log_line(message: str, total_log: TextIO | None=None) -> None:
    print(message, flush=True)
    if total_log is not None:
        total_log.write(message + '\n')
        total_log.flush()

def split_files(converted_root: Path, scenarios: list[str], folds: list[str]) -> list[Path]:
    files: list[Path] = []
    for scenario in scenarios:
        for fold in folds:
            fold_dir = converted_root / scenario / fold
            for name in ('train.csv', 'valid.csv', 'test.csv'):
                path = fold_dir / name
                if name == 'valid.csv' and (not path.exists()):
                    path = fold_dir / 'validation.csv'
                if not path.exists():
                    raise FileNotFoundError(f'Converted split file not found: {path}')
                files.append(path)
    return files

def run_command(command: list[str], log_path: Path | None=None, env: dict | None=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    log_context = open(log_path, 'w', encoding='utf-8') if log_path else nullcontext(None)
    with log_context as log_f:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            if log_f is not None:
                log_f.write(line)
                log_f.flush()
        return_code = proc.wait()
    if return_code != 0:
        target = f' See {log_path}' if log_path else ''
        raise RuntimeError(f'Command failed with exit code {return_code}.{target}')

def extract_embeddings(csv_paths: list[Path], drug_embedding_dir: Path, protein_embedding_dir: Path, drug_model: str, protein_model: str, drug_batch_size: int, protein_batch_size: int, output_dir: Path) -> None:
    common = [sys.executable, '-u']
    csv_args = [str(path) for path in csv_paths]
    run_command(common + ['features.py', 'drug', '--csv', *csv_args, '--out_dir', str(drug_embedding_dir), '--model_name', drug_model, '--batch_size', str(drug_batch_size)], output_dir / 'extract_drug_embeddings.log')
    run_command(common + ['features.py', 'protein', '--csv', *csv_args, '--out_dir', str(protein_embedding_dir), '--model_name', protein_model, '--batch_size', str(protein_batch_size)], output_dir / 'extract_protein_embeddings.log')

def run_one_fold(base_cfg: dict, split_dir: Path, fold_out_dir: Path, scenario: str, fold: str, dataset_name: str, total_log: TextIO | None=None) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg['data']['dataset_name'] = dataset_name
    cfg['data']['split_type'] = scenario
    cfg['data']['split_dir'] = str(split_dir).replace('\\', '/')
    cfg['seed'] = int(base_cfg.get('seed', 42)) + int(fold)
    fold_out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = fold_out_dir / 'config.yaml'
    log_path = fold_out_dir / 'train.log'
    save_config(cfg, cfg_path)
    command = [sys.executable, '-u', 'main.py', 'train', '--config', str(cfg_path), '--output_dir', str(fold_out_dir)]
    started_at = now_text()
    start = time.perf_counter()
    best_threshold: dict | None = None
    test_metrics: dict | None = None
    log_line(f'========== START {scenario}/{fold} at {started_at} ==========', total_log)
    log_line(f'split_dir={split_dir}', total_log)
    log_line(f'output_dir={fold_out_dir}', total_log)
    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f'========== START {scenario}/{fold} at {started_at} ==========\n')
        log_f.write(f"command={' '.join(command)}\n")
        log_f.flush()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log_f.write(line)
            log_f.flush()
            if total_log is not None:
                total_log.write(line)
                total_log.flush()
            parsed_threshold = parse_json_line(line, 'best_threshold ')
            if parsed_threshold is not None:
                best_threshold = parsed_threshold
            parsed_test = parse_json_line(line, 'test ')
            if parsed_test is not None:
                test_metrics = parsed_test
        return_code = proc.wait()
    elapsed = time.perf_counter() - start
    ended_at = now_text()
    if return_code != 0:
        raise RuntimeError(f'{scenario}/{fold} failed with exit code {return_code}. See {log_path}')
    if test_metrics is None:
        raise RuntimeError(f'{scenario}/{fold} did not emit test metrics. See {log_path}')
    result = {'scenario': scenario, 'fold': fold, 'split_dir': str(split_dir).replace('\\', '/'), 'output_dir': str(fold_out_dir).replace('\\', '/'), 'started_at': started_at, 'ended_at': ended_at, 'time_seconds': elapsed, 'time_minutes': elapsed / 60.0, 'best_threshold': best_threshold, 'test': test_metrics}
    with open(fold_out_dir / 'fold_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    log_line(f'{scenario}/{fold} finished in {elapsed:.2f}s ({elapsed / 60.0:.2f}min), test={json.dumps(test_metrics, ensure_ascii=False)}', total_log)
    log_line(f'========== END {scenario}/{fold} at {ended_at} ==========', total_log)
    return result

def summarize(results: list[dict], total_seconds: float) -> dict:
    scenario_summaries = {}
    for scenario in sorted({result['scenario'] for result in results}):
        scenario_results = [result for result in results if result['scenario'] == scenario]
        scenario_summaries[scenario] = summarize_metric_block(scenario_results)
    return {'folds': results, 'by_scenario': scenario_summaries, 'overall': summarize_metric_block(results), 'time': {'fold_seconds': [float(result['time_seconds']) for result in results], 'fold_minutes': [float(result['time_minutes']) for result in results], 'total_seconds': float(total_seconds), 'total_minutes': float(total_seconds / 60.0), 'mean_fold_seconds': float(np.mean([result['time_seconds'] for result in results])) if results else 0.0, 'std_fold_seconds': float(np.std([result['time_seconds'] for result in results], ddof=1)) if len(results) > 1 else 0.0, 'mean_fold_minutes': float(np.mean([result['time_minutes'] for result in results])) if results else 0.0, 'std_fold_minutes': float(np.std([result['time_minutes'] for result in results], ddof=1)) if len(results) > 1 else 0.0}}

def summarize_metric_block(results: list[dict]) -> dict:
    metric_names = sorted({key for result in results for key, value in result['test'].items() if isinstance(value, int | float)})
    metrics = {}
    for name in metric_names:
        values = np.array([float(result['test'][name]) for result in results], dtype=np.float64)
        metrics[name] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 'values': values.tolist()}
    return metrics

def write_summary_csv(summary: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'metric', 'mean', 'std', 'values'])
        for group, block in summary['by_scenario'].items():
            for metric, stats in block.items():
                writer.writerow([group, metric, stats['mean'], stats['std'], json.dumps(stats['values'])])
        for metric, stats in summary['overall'].items():
            writer.writerow(['overall', metric, stats['mean'], stats['std'], json.dumps(stats['values'])])
        writer.writerow(['time', 'total_minutes', summary['time']['total_minutes'], '', ''])
        writer.writerow(['time', 'mean_fold_minutes', summary['time']['mean_fold_minutes'], '', ''])
        writer.writerow(['time', 'std_fold_minutes', summary['time']['std_fold_minutes'], '', ''])

def write_fold_results_csv(results: list[dict], path: Path) -> None:
    metric_names = sorted({key for result in results for key, value in result['test'].items() if isinstance(value, int | float)})
    threshold_names = sorted({key for result in results if isinstance(result.get('best_threshold'), dict) for key, value in result['best_threshold'].items() if isinstance(value, int | float)})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'fold', 'started_at', 'ended_at', 'time_seconds', 'time_minutes', 'split_dir', 'output_dir', *[f'test_{name}' for name in metric_names], *[f'best_threshold_{name}' for name in threshold_names]])
        for result in results:
            best_threshold = result.get('best_threshold') or {}
            writer.writerow([result['scenario'], result['fold'], result['started_at'], result['ended_at'], result['time_seconds'], result['time_minutes'], result['split_dir'], result['output_dir'], *[result['test'].get(name, '') for name in metric_names], *[best_threshold.get(name, '') for name in threshold_names]])

def run_cli() -> None:
    parser = argparse.ArgumentParser(description='Prepare and run CS-LSDA-DTI experiments on data1 datasets.')
    parser.add_argument('--input_root', help='Raw dataset root containing random/e2/e3/e4.')
    parser.add_argument('--converted_root', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--base_config', default='config.yaml')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scenarios', nargs='+', default=DEFAULT_SCENARIOS)
    parser.add_argument('--folds', nargs='+', default=[str(i) for i in range(10)])
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--drug_embedding_dir', required=True)
    parser.add_argument('--protein_embedding_dir', required=True)
    parser.add_argument('--drug_model', default='molformer')
    parser.add_argument('--protein_model', default='facebook/esm2_t30_150M_UR50D')
    parser.add_argument('--drug_batch_size', type=int, default=16)
    parser.add_argument('--protein_batch_size', type=int, default=2)
    args = parser.parse_args()
    if not (args.convert or args.extract or args.train):
        args.convert = True
        args.extract = True
        args.train = True
    converted_root = Path(args.converted_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.convert:
        if not args.input_root:
            parser.error('--input_root is required with --convert')
        convert_dataset(input_root=Path(args.input_root), output_root=converted_root, scenarios=args.scenarios, folds=args.folds)
    if args.extract:
        csv_paths = split_files(converted_root, args.scenarios, args.folds)
        extract_embeddings(csv_paths=csv_paths, drug_embedding_dir=Path(args.drug_embedding_dir), protein_embedding_dir=Path(args.protein_embedding_dir), drug_model=args.drug_model, protein_model=args.protein_model, drug_batch_size=args.drug_batch_size, protein_batch_size=args.protein_batch_size, output_dir=output_dir)
    if args.train:
        base_cfg = load_config(args.base_config)
        base_cfg['embedding']['drug_embedding_dir'] = args.drug_embedding_dir
        base_cfg['embedding']['protein_embedding_dir'] = args.protein_embedding_dir
        all_start = time.perf_counter()
        results = []
        total_log_path = output_dir / 'total.log'
        with open(total_log_path, 'w', encoding='utf-8') as total_log:
            log_line(f'========== RUN START {now_text()} ==========', total_log)
            log_line(f'base_config={args.base_config}', total_log)
            log_line(f'converted_root={converted_root}', total_log)
            log_line(f"scenarios={' '.join(args.scenarios)}", total_log)
            log_line(f"folds={' '.join(args.folds)}", total_log)
            log_line(f'drug_embedding_dir={args.drug_embedding_dir}', total_log)
            log_line(f'protein_embedding_dir={args.protein_embedding_dir}', total_log)
            for scenario in args.scenarios:
                for fold in args.folds:
                    split_dir = converted_root / scenario / fold
                    fold_out_dir = output_dir / scenario / fold
                    log_line(f'========== {scenario}/{fold}: {split_dir} ==========', total_log)
                    results.append(run_one_fold(base_cfg, split_dir, fold_out_dir, scenario, fold, args.dataset_name, total_log))
            total_seconds = time.perf_counter() - all_start
            summary = summarize(results, total_seconds)
            with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            write_summary_csv(summary, output_dir / 'summary.csv')
            write_fold_results_csv(results, output_dir / 'fold_results.csv')
            log_line('========== summary ==========', total_log)
            for scenario, block in summary['by_scenario'].items():
                log_line(f'[{scenario}]', total_log)
                for metric, stats in block.items():
                    log_line(f"{metric}: {stats['mean']:.8f} +/- {stats['std']:.8f}", total_log)
            log_line(f"mean_fold_time: {summary['time']['mean_fold_seconds']:.2f}s ({summary['time']['mean_fold_minutes']:.2f}min)", total_log)
            log_line(f"std_fold_time: {summary['time']['std_fold_seconds']:.2f}s ({summary['time']['std_fold_minutes']:.2f}min)", total_log)
            log_line(f'total_time: {total_seconds:.2f}s ({total_seconds / 60.0:.2f}min)', total_log)
            log_line(f'========== RUN END {now_text()} ==========', total_log)
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def existing_optional_float(value) -> float | None:
    if value is None or value == '':
        return None
    return float(value)

def existing_load_results(root: Path, scenario: str, folds: list[str]) -> list[dict]:
    results = []
    missing = []
    for fold in folds:
        path = root / scenario / fold / 'fold_result.json'
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        results.append(result)
    if missing:
        raise FileNotFoundError('Missing fold_result.json files:\n' + '\n'.join(missing))
    return results

def existing_numeric_metric_names(results: list[dict]) -> list[str]:
    return sorted({key for result in results for key, value in result.get('test', {}).items() if isinstance(value, int | float)})

def existing_numeric_threshold_names(results: list[dict]) -> list[str]:
    return sorted({key for result in results if isinstance(result.get('best_threshold'), dict) for key, value in result['best_threshold'].items() if isinstance(value, int | float)})

def existing_summarize(results: list[dict]) -> dict:
    metrics = {}
    for name in existing_numeric_metric_names(results):
        values = np.array([float(result['test'][name]) for result in results], dtype=np.float64)
        metrics[name] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 'values': values.tolist()}
    known_times = [value for value in (existing_optional_float(result.get('time_seconds')) for result in results) if value is not None]
    times = np.array(known_times, dtype=np.float64)
    return {'folds': results, 'metrics': metrics, 'time': {'fold_seconds': [existing_optional_float(result.get('time_seconds')) for result in results], 'fold_minutes': [existing_optional_float(result.get('time_minutes')) for result in results], 'known_time_folds': int(len(times)), 'missing_time_folds': int(len(results) - len(times)), 'total_seconds': float(np.sum(times)), 'total_minutes': float(np.sum(times) / 60.0), 'mean_fold_seconds': float(np.mean(times)) if len(times) else 0.0, 'std_fold_seconds': float(np.std(times, ddof=1)) if len(times) > 1 else 0.0, 'mean_fold_minutes': float(np.mean(times / 60.0)) if len(times) else 0.0, 'std_fold_minutes': float(np.std(times / 60.0, ddof=1)) if len(times) > 1 else 0.0}}

def existing_write_summary_csv(summary: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'values'])
        for metric, stats in summary['metrics'].items():
            writer.writerow([metric, stats['mean'], stats['std'], json.dumps(stats['values'])])
        writer.writerow(['total_minutes', summary['time']['total_minutes'], '', ''])
        writer.writerow(['mean_fold_minutes', summary['time']['mean_fold_minutes'], '', ''])
        writer.writerow(['std_fold_minutes', summary['time']['std_fold_minutes'], '', ''])

def existing_write_fold_results_csv(results: list[dict], path: Path) -> None:
    metric_names = existing_numeric_metric_names(results)
    threshold_names = existing_numeric_threshold_names(results)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'fold', 'started_at', 'ended_at', 'time_seconds', 'time_minutes', 'split_dir', 'output_dir', *[f'test_{name}' for name in metric_names], *[f'best_threshold_{name}' for name in threshold_names]])
        for result in results:
            best_threshold = result.get('best_threshold') or {}
            writer.writerow([result.get('scenario', ''), result.get('fold', ''), result.get('started_at', ''), result.get('ended_at', ''), result.get('time_seconds', ''), result.get('time_minutes', ''), result.get('split_dir', ''), result.get('output_dir', ''), *[result.get('test', {}).get(name, '') for name in metric_names], *[best_threshold.get(name, '') for name in threshold_names]])

def existing_main() -> None:
    parser = argparse.ArgumentParser(description='Summarize existing fold_result.json files.')
    parser.add_argument('--root', required=True, help='Output root, e.g. outputs/biosnap_data1_random_1-runs')
    parser.add_argument('--scenario', required=True, help='Scenario name, e.g. random/e2/e3/e4')
    parser.add_argument('--folds', nargs='+', default=[str(i) for i in range(10)])
    parser.add_argument('--prefix', default='summary_10fold_fixed')
    args = parser.parse_args()
    root = Path(args.root)
    results = existing_load_results(root, args.scenario, args.folds)
    summary = existing_summarize(results)
    summary_json = root / f'{args.prefix}.json'
    summary_csv = root / f'{args.prefix}.csv'
    fold_csv = root / f'{args.prefix}_fold_results.csv'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    existing_write_summary_csv(summary, summary_csv)
    existing_write_fold_results_csv(results, fold_csv)
    print(f'Loaded {len(results)} folds from {root / args.scenario}')
    for metric, stats in summary['metrics'].items():
        print(f"{metric}: {stats['mean']:.8f} +/- {stats['std']:.8f}")
    print(f"total_time: {summary['time']['total_seconds']:.2f}s ({summary['time']['total_minutes']:.2f}min)")
    print(f'Wrote {summary_json}')
    print(f'Wrote {summary_csv}')
    print(f'Wrote {fold_csv}')

if __name__ == "__main__":
    commands = {
        "train": train_cli,
        "evaluate": evaluate_cli,
        "run": run_cli,
        "summarize": existing_main,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python main.py {train|evaluate|run|summarize} [arguments]")
    command = commands[sys.argv.pop(1)]
    command()
