from __future__ import annotations

from data import read_dti_csv

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
MODEL_ALIASES = {'molformer': 'ibm/MoLFormer-XL-both-10pct', 'chemberta': 'seyonec/ChemBERTa-zinc-base-v1'}
MOLFORMER_REVISION = 'a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8'

def expand_csv_inputs(inputs: list[str]) -> list[str]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob('*.csv'))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f'CSV input does not exist: {path}')
    selected = sorted({path.resolve() for path in paths if path.name != 'conversion_summary.csv'})
    if not selected:
        raise FileNotFoundError('No split CSV files were found.')
    return [str(path) for path in selected]

def removable_special_token_ids(tokenizer) -> set[int]:
    names = ['bos_token_id', 'eos_token_id', 'cls_token_id', 'sep_token_id', 'pad_token_id']
    return {getattr(tokenizer, name) for name in names if getattr(tokenizer, name) is not None}

def load_unique_drugs(csv_paths: list[str]) -> pd.DataFrame:
    frames = [read_dti_csv(path)[['drug_id', 'smiles']] for path in csv_paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates('drug_id')
    return df.reset_index(drop=True)

def validate_and_write_manifest(out_dir: Path, manifest: dict, overwrite: bool) -> None:
    path = out_dir / 'embedding_manifest.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        keys = ('model_id', 'revision', 'max_length', 'hidden_size')
        mismatches = {key: (existing.get(key), manifest.get(key)) for key in keys if existing.get(key) != manifest.get(key)}
        if mismatches and (not overwrite):
            raise RuntimeError(f'Embedding cache metadata differs in {path}: {mismatches}. Use a new output directory or pass --overwrite.')
    elif any(out_dir.glob('*.pt')):
        first = next(out_dir.glob('*.pt'))
        cached_dim = int(torch.load(first, map_location='cpu').shape[-1])
        if cached_dim != manifest['hidden_size']:
            raise RuntimeError(f"Unverified cache dimension {cached_dim} does not match model dimension {manifest['hidden_size']}: {out_dir}")
        print(f'Warning: adopting legacy cache without a manifest: {out_dir}')
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temp.replace(path)

def drug_cli() -> None:
    parser = argparse.ArgumentParser(description='Extract frozen drug token embeddings.')
    parser.add_argument('--csv', nargs='+', required=True, help='Split CSV files or directories containing them.')
    parser.add_argument('--out_dir', default='embeddings/biosnap/drug_molformer')
    parser.add_argument('--model_name', default='molformer', help='molformer, chemberta, or HF id.')
    parser.add_argument('--revision', default=None, help="Hugging Face revision. The molformer alias uses the repository's pinned revision.")
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    model_id = MODEL_ALIASES.get(args.model_name, args.model_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = expand_csv_inputs(args.csv)
    print(f'CSV file count: {len(csv_paths)}')
    drugs = load_unique_drugs(csv_paths)
    revision_arg = args.revision
    if revision_arg is None and args.model_name == 'molformer':
        revision_arg = MOLFORMER_REVISION
    model_kwargs = {'trust_remote_code': True}
    if revision_arg:
        model_kwargs['revision'] = revision_arg
    tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
    model = AutoModel.from_pretrained(model_id, **model_kwargs).to(args.device)
    model.eval()
    revision = revision_arg or getattr(model.config, '_commit_hash', None)
    validate_and_write_manifest(out_dir, {'model_id': model_id, 'revision': revision, 'max_length': args.max_length, 'hidden_size': int(model.config.hidden_size), 'special_tokens_removed': True}, args.overwrite)
    with torch.no_grad():
        for start in tqdm(range(0, len(drugs), args.batch_size), desc='drug embeddings'):
            batch = drugs.iloc[start:start + args.batch_size]
            todo = [(str(row.drug_id), str(row.smiles)) for row in batch.itertuples(index=False) if args.overwrite or not (out_dir / f'{row.drug_id}.pt').exists()]
            if not todo:
                continue
            drug_ids, smiles = zip(*todo)
            encoded = tokenizer(list(smiles), padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state.detach().cpu()
            attention_mask = encoded['attention_mask'].detach().cpu().bool()
            special_ids = removable_special_token_ids(tokenizer)
            input_ids = encoded['input_ids'].detach().cpu()
            for i, drug_id in enumerate(drug_ids):
                keep = attention_mask[i].clone()
                for special_id in special_ids:
                    keep &= input_ids[i] != special_id
                tokens = hidden[i, keep][:args.max_length].contiguous()
                if tokens.numel() == 0:
                    tokens = hidden[i, attention_mask[i]][:args.max_length].contiguous()
                torch.save(tokens, out_dir / f'{drug_id}.pt')
    print(f'Saved drug embeddings to {out_dir}')

def load_unique_targets(csv_paths: list[str]) -> pd.DataFrame:
    frames = [read_dti_csv(path)[['target_id', 'protein_sequence']] for path in csv_paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates('target_id')
    return df.reset_index(drop=True)

def protein_cli() -> None:
    parser = argparse.ArgumentParser(description='Extract frozen ESM-2 residue embeddings.')
    parser.add_argument('--csv', nargs='+', required=True, help='Split CSV files or directories containing them.')
    parser.add_argument('--out_dir', default='embeddings/biosnap/protein_esm2_150m')
    parser.add_argument('--model_name', default='facebook/esm2_t30_150M_UR50D')
    parser.add_argument('--revision', default=None, help='Optional pinned Hugging Face revision.')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = expand_csv_inputs(args.csv)
    print(f'CSV file count: {len(csv_paths)}')
    targets = load_unique_targets(csv_paths)
    model_kwargs = {'revision': args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **model_kwargs)
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(args.device)
    model.eval()
    revision = args.revision or getattr(model.config, '_commit_hash', None)
    validate_and_write_manifest(out_dir, {'model_id': args.model_name, 'revision': revision, 'max_length': args.max_length, 'hidden_size': int(model.config.hidden_size), 'special_tokens_removed': True}, args.overwrite)
    tokenizer_max_length = args.max_length + 2
    with torch.no_grad():
        for start in tqdm(range(0, len(targets), args.batch_size), desc='protein embeddings'):
            batch = targets.iloc[start:start + args.batch_size]
            todo = [(str(row.target_id), str(row.protein_sequence)[:args.max_length]) for row in batch.itertuples(index=False) if args.overwrite or not (out_dir / f'{row.target_id}.pt').exists()]
            if not todo:
                continue
            target_ids, sequences = zip(*todo)
            encoded = tokenizer(list(sequences), padding=True, truncation=True, max_length=tokenizer_max_length, return_tensors='pt')
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state.detach().cpu()
            attention_mask = encoded['attention_mask'].detach().cpu().bool()
            input_ids = encoded['input_ids'].detach().cpu()
            special_ids = removable_special_token_ids(tokenizer)
            for i, target_id in enumerate(target_ids):
                keep = attention_mask[i].clone()
                for special_id in special_ids:
                    keep &= input_ids[i] != special_id
                residues = hidden[i, keep][:args.max_length].contiguous()
                torch.save(residues, out_dir / f'{target_id}.pt')
    print(f'Saved protein embeddings to {out_dir}')

if __name__ == "__main__":
    import sys
    commands = {"drug": drug_cli, "protein": protein_cli}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python features.py {drug|protein} [arguments]")
    command = commands[sys.argv.pop(1)]
    command()
