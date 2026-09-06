from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd
STANDARD_COLUMNS = ['drug_id', 'target_id', 'smiles', 'protein_sequence', 'label']
DATA1_COLUMNS = {'SMILES', 'Protein', 'Y'}

def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}_{digest}'

def normalize_dti_dataframe(df: pd.DataFrame, source: str | Path='<dataframe>') -> pd.DataFrame:
    if set(STANDARD_COLUMNS).issubset(df.columns):
        return df[STANDARD_COLUMNS].copy()
    if DATA1_COLUMNS.issubset(df.columns):
        smiles = df['SMILES'].astype(str)
        proteins = df['Protein'].astype(str)
        return pd.DataFrame({'drug_id': smiles.map(lambda value: stable_id('D', value)), 'target_id': proteins.map(lambda value: stable_id('T', value)), 'smiles': smiles, 'protein_sequence': proteins, 'label': df['Y'].astype(float)})
    missing_standard = sorted(set(STANDARD_COLUMNS) - set(df.columns))
    missing_data1 = sorted(DATA1_COLUMNS - set(df.columns))
    raise ValueError(f'{source} does not match a supported DTI CSV schema. Missing standard columns: {missing_standard}; missing data1 columns: {missing_data1}')

def read_dti_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_dti_dataframe(df, source=path)
import torch
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PAD_INDEX = 0
AA_UNK_INDEX = 21
AA_VOCAB_SIZE = 22
AA_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}
SMILES_CHARACTERS = ''.join((chr(code) for code in range(33, 127)))
SMILES_TO_INDEX = {character: index + 2 for index, character in enumerate(SMILES_CHARACTERS)}
SMILES_PAD_INDEX = 0
SMILES_UNK_INDEX = 1
SMILES_VOCAB_SIZE = len(SMILES_CHARACTERS) + 2

def encode_amino_acids(sequence: str, max_length: int) -> torch.Tensor:
    indices = [AA_TO_INDEX.get(residue.upper(), AA_UNK_INDEX) for residue in sequence[:max_length]]
    return torch.tensor(indices or [AA_UNK_INDEX], dtype=torch.long)

def encode_smiles_characters(smiles: str, max_length: int) -> torch.Tensor:
    indices = [SMILES_TO_INDEX.get(character, SMILES_UNK_INDEX) for character in smiles[:max_length]]
    return torch.tensor(indices or [SMILES_UNK_INDEX], dtype=torch.long)
import torch
ATOM_FEATURE_DIM = 128
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AMINO_ACID_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}

def encode_protein_sequence(sequence: str, max_length: int) -> torch.Tensor:
    """Encode a protein sequence with 0 reserved for padding/unknown residues."""
    indices = [AMINO_ACID_TO_INDEX.get(residue, 0) for residue in sequence[:max_length]]
    return torch.tensor(indices or [0], dtype=torch.long)

def smiles_to_graph(smiles: str, max_atoms: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact atom and bond tensors from a SMILES string.

    RDKit is intentionally imported only here so embedding-only experiments do
    not require it. Atom features are a stable, fixed-size encoding and bond
    features use four bond categories plus aromaticity.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError('RDKit is required for model.use_multiview_residual=true. Install it with `conda install -c conda-forge rdkit`.') from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Invalid SMILES encountered: {smiles!r}')
    atom_count = min(molecule.GetNumAtoms(), max_atoms)
    node_features = torch.zeros((atom_count, ATOM_FEATURE_DIM), dtype=torch.float32)
    adjacency = torch.zeros((atom_count, atom_count), dtype=torch.bool)
    edge_features = torch.zeros((atom_count, atom_count, 5), dtype=torch.float32)
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        if atom_index >= atom_count:
            break
        atomic_number = min(max(atom.GetAtomicNum(), 1), 118)
        node_features[atom_index, atomic_number - 1] = 1.0
        node_features[atom_index, 118] = atom.GetDegree() / 4.0
        node_features[atom_index, 119] = atom.GetFormalCharge() / 4.0
        node_features[atom_index, 120] = float(atom.GetIsAromatic())
        node_features[atom_index, 121] = atom.GetTotalNumHs() / 4.0
        node_features[atom_index, 122] = float(atom.IsInRing())
    bond_to_index = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    for bond in molecule.GetBonds():
        source, target = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if source >= atom_count or target >= atom_count:
            continue
        bond_features = torch.zeros(5, dtype=torch.float32)
        bond_features[bond_to_index.get(bond.GetBondType(), 0)] = 1.0
        bond_features[4] = float(bond.GetIsAromatic())
        adjacency[source, target] = True
        adjacency[target, source] = True
        edge_features[source, target] = bond_features
        edge_features[target, source] = bond_features
    return (node_features, adjacency, edge_features)
from pathlib import Path
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Dataset backed by precomputed per-drug and per-target token embeddings."""

    def __init__(self, csv_path: str | Path, drug_embedding_dir: str | Path, protein_embedding_dir: str | Path, max_drug_len: int=128, max_protein_len: int=1024, use_multiview_residual: bool=False, max_graph_atoms: int=128, max_raw_protein_len: int=1024, drug_encoder_type: str='plm', protein_encoder_type: str='plm') -> None:
        self.df = read_dti_csv(csv_path)
        self.drug_embedding_dir = Path(drug_embedding_dir)
        self.protein_embedding_dir = Path(protein_embedding_dir)
        self.max_drug_len = max_drug_len
        self.max_protein_len = max_protein_len
        self.use_multiview_residual = use_multiview_residual
        self.max_graph_atoms = max_graph_atoms
        self.max_raw_protein_len = max_raw_protein_len
        self.drug_encoder_type = drug_encoder_type
        self.protein_encoder_type = protein_encoder_type
        self._graph_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._smiles_token_cache: dict[str, torch.Tensor] = {}
        self._protein_token_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        drug_id = str(row['drug_id'])
        target_id = str(row['target_id'])
        drug = torch.load(self.drug_embedding_dir / f'{drug_id}.pt', map_location='cpu') if self.drug_encoder_type == 'plm' else torch.zeros((1, 1), dtype=torch.float32)
        protein = torch.load(self.protein_embedding_dir / f'{target_id}.pt', map_location='cpu') if self.protein_encoder_type == 'plm' else torch.zeros((1, 1), dtype=torch.float32)
        drug = drug[:self.max_drug_len].float()
        protein = protein[:self.max_protein_len].float()
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        item: dict[str, torch.Tensor | str] = {'drug_id': drug_id, 'target_id': target_id, 'drug_embedding': drug, 'protein_embedding': protein, 'label': label}
        if self.drug_encoder_type == 'smiles_cnn':
            if drug_id not in self._smiles_token_cache:
                self._smiles_token_cache[drug_id] = encode_smiles_characters(str(row['smiles']), self.max_drug_len)
            item['raw_smiles_tokens'] = self._smiles_token_cache[drug_id]
        if self.protein_encoder_type == 'aa_cnn':
            if target_id not in self._protein_token_cache:
                self._protein_token_cache[target_id] = encode_amino_acids(str(row['protein_sequence']), self.max_protein_len)
            item['raw_protein_tokens'] = self._protein_token_cache[target_id]
        if self.use_multiview_residual:
            smiles = str(row['smiles'])
            if drug_id not in self._graph_cache:
                self._graph_cache[drug_id] = smiles_to_graph(smiles, self.max_graph_atoms)
            node_features, adjacency, edge_features = self._graph_cache[drug_id]
            item.update({'graph_node_features': node_features, 'graph_adjacency': adjacency, 'graph_edge_features': edge_features, 'protein_sequence': encode_protein_sequence(str(row['protein_sequence']), self.max_raw_protein_len)})
        return item
import torch

def _pad_tokens(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(tensors)
    max_len = max((t.size(0) for t in tensors))
    dim = tensors[0].size(-1)
    padded = tensors[0].new_zeros((batch, max_len, dim))
    mask = torch.ones((batch, max_len), dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        length = tensor.size(0)
        padded[i, :length] = tensor
        mask[i, :length] = False
    return (padded, mask)

def _pad_indices(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(tensors)
    max_len = max((tensor.size(0) for tensor in tensors))
    padded = torch.zeros((batch, max_len), dtype=torch.long)
    mask = torch.ones((batch, max_len), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        length = tensor.size(0)
        padded[index, :length] = tensor
        mask[index, :length] = False
    return (padded, mask)

def _pad_graphs(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_nodes = max((item['graph_node_features'].size(0) for item in batch))
    batch_size = len(batch)
    node_dim = batch[0]['graph_node_features'].size(-1)
    edge_dim = batch[0]['graph_edge_features'].size(-1)
    nodes = torch.zeros((batch_size, max_nodes, node_dim), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    edges = torch.zeros((batch_size, max_nodes, max_nodes, edge_dim), dtype=torch.float32)
    node_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item['graph_node_features'].size(0)
        nodes[index, :count] = item['graph_node_features']
        adjacency[index, :count, :count] = item['graph_adjacency']
        edges[index, :count, :count] = item['graph_edge_features']
        node_mask[index, :count] = False
    return {'graph_node_features': nodes, 'graph_adjacency': adjacency, 'graph_edge_features': edges, 'graph_node_mask': node_mask}

def embedding_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    drug, drug_mask = _pad_tokens([item['drug_embedding'] for item in batch])
    protein, protein_mask = _pad_tokens([item['protein_embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    result: dict[str, torch.Tensor | list[str]] = {'drug_ids': [item['drug_id'] for item in batch], 'target_ids': [item['target_id'] for item in batch], 'drug_embedding': drug, 'protein_embedding': protein, 'drug_mask': drug_mask, 'protein_mask': protein_mask, 'label': labels}
    if 'raw_smiles_tokens' in batch[0]:
        raw_smiles, raw_smiles_mask = _pad_indices([item['raw_smiles_tokens'] for item in batch])
        result['raw_smiles_tokens'] = raw_smiles
        result['raw_smiles_mask'] = raw_smiles_mask
    if 'raw_protein_tokens' in batch[0]:
        raw_protein, raw_protein_mask = _pad_indices([item['raw_protein_tokens'] for item in batch])
        result['raw_protein_tokens'] = raw_protein
        result['raw_protein_mask'] = raw_protein_mask
    if 'graph_node_features' in batch[0]:
        result.update(_pad_graphs(batch))
        protein_sequence, protein_sequence_mask = _pad_tokens([item['protein_sequence'].unsqueeze(-1).float() for item in batch])
        result['protein_sequence'] = protein_sequence.squeeze(-1).long()
        result['protein_sequence_mask'] = protein_sequence_mask
    return result
import argparse
from pathlib import Path
import pandas as pd
SPLIT_FILE_MAP = {'train': ('train.csv', 'train.csv'), 'validation': ('validation.csv', 'valid.csv'), 'test': ('test.csv', 'test.csv')}

def make_ids(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create entity IDs from content, never from clustering assignments."""
    drug_ids = df['SMILES'].astype(str).map(lambda value: stable_id('D', value))
    target_ids = df['Protein'].astype(str).map(lambda value: stable_id('T', value))
    return (drug_ids, target_ids)

def convert_frame(df: pd.DataFrame, source: str | Path) -> tuple[pd.DataFrame, int]:
    required = {'SMILES', 'Protein', 'Y'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{source} is missing required columns: {missing}')
    drug_ids, target_ids = make_ids(df)
    out = pd.DataFrame({'drug_id': drug_ids, 'target_id': target_ids, 'smiles': df['SMILES'].astype(str), 'protein_sequence': df['Protein'].astype(str), 'label': pd.to_numeric(df['Y'], errors='raise').astype(float)})
    if out[['smiles', 'protein_sequence', 'label']].isna().any().any():
        raise ValueError(f'{source} contains missing SMILES, protein sequences, or labels')
    labels = set(out['label'].unique())
    if not labels.issubset({0.0, 1.0}):
        raise ValueError(f'{source} contains non-binary labels: {sorted(labels)}')
    conflicts = out.groupby(['drug_id', 'target_id'])['label'].nunique()
    if (conflicts > 1).any():
        raise ValueError(f'{source} contains drug-target pairs with conflicting labels')
    before = len(out)
    out = out.drop_duplicates(['drug_id', 'target_id'], keep='first').reset_index(drop=True)
    return (out, before - len(out))

def convert_file(input_path: Path, output_path: Path, id_strategy: str='hash') -> tuple[int, int, int]:
    if id_strategy != 'hash':
        raise ValueError('Only stable hash IDs are supported')
    df = pd.read_csv(input_path)
    out, _ = convert_frame(df, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return (len(out), out['drug_id'].nunique(), out['target_id'].nunique())

def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame['drug_id'], frame['target_id']))

def remove_pairs(frame: pd.DataFrame, excluded: set[tuple[str, str]]) -> tuple[pd.DataFrame, int]:
    if not excluded:
        return (frame, 0)
    keep = [pair not in excluded for pair in zip(frame['drug_id'], frame['target_id'])]
    cleaned = frame.loc[keep].reset_index(drop=True)
    return (cleaned, len(frame) - len(cleaned))

def validate_cold_split(scenario: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared_drugs = set(train['drug_id']) & set(test['drug_id'])
    shared_targets = set(train['target_id']) & set(test['target_id'])
    if scenario == 'e2' and shared_drugs:
        raise ValueError(f'E2 split has {len(shared_drugs)} test drugs present in training')
    if scenario == 'e3' and shared_targets:
        raise ValueError(f'E3 split has {len(shared_targets)} test targets present in training')
    if scenario == 'e4' and (shared_drugs or shared_targets):
        raise ValueError(f'E4 split violates double-cold separation: shared_drugs={len(shared_drugs)}, shared_targets={len(shared_targets)}')

def convert_dataset(input_root: Path, output_root: Path, scenarios: list[str], folds: list[str] | None) -> None:
    rows = []
    for scenario in scenarios:
        scenario_dir = input_root / scenario
        if not scenario_dir.exists():
            raise FileNotFoundError(f'Scenario directory not found: {scenario_dir}')
        if folds is None:
            fold_dirs = sorted([path for path in scenario_dir.iterdir() if path.is_dir()], key=lambda path: int(path.name) if path.name.isdigit() else path.name)
        else:
            fold_dirs = [scenario_dir / fold for fold in folds]
        if not fold_dirs:
            raise FileNotFoundError(f'No fold directories found in {scenario_dir}')
        for fold_dir in fold_dirs:
            if not fold_dir.exists():
                raise FileNotFoundError(f'Fold directory not found: {fold_dir}')
            converted: dict[str, pd.DataFrame] = {}
            duplicate_counts: dict[str, int] = {}
            for source_name, (input_name, target_name) in SPLIT_FILE_MAP.items():
                input_path = fold_dir / input_name
                if source_name == 'validation' and (not input_path.exists()):
                    input_path = fold_dir / 'valid.csv'
                if not input_path.exists():
                    raise FileNotFoundError(f'Split file not found: {input_path}')
                converted[source_name], duplicate_counts[source_name] = convert_frame(pd.read_csv(input_path), input_path)
            converted['validation'], removed_valid = remove_pairs(converted['validation'], pair_set(converted['test']))
            converted['train'], removed_train = remove_pairs(converted['train'], pair_set(converted['validation']) | pair_set(converted['test']))
            cross_removed = {'train': removed_train, 'validation': removed_valid, 'test': 0}
            validate_cold_split(scenario, converted['train'], converted['test'])
            for source_name, (_, target_name) in SPLIT_FILE_MAP.items():
                output_path = output_root / scenario / fold_dir.name / target_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frame = converted[source_name]
                frame.to_csv(output_path, index=False)
                rows.append({'scenario': scenario, 'fold': fold_dir.name, 'split': target_name.removesuffix('.csv'), 'rows': len(frame), 'drugs': frame['drug_id'].nunique(), 'targets': frame['target_id'].nunique(), 'within_split_duplicates_removed': duplicate_counts[source_name], 'cross_split_duplicates_removed': cross_removed[source_name], 'id_strategy': 'sha1_content_hash', 'path': str(output_path).replace('\\', '/')})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / 'conversion_summary.csv', index=False)
    print(f'Converted DTI splits to {output_root}')
    print(summary[['scenario', 'fold', 'split', 'rows', 'drugs', 'targets', 'within_split_duplicates_removed', 'cross_split_duplicates_removed']].to_string(index=False))

def convert_cli() -> None:
    parser = argparse.ArgumentParser(description='Convert random/E2/E3/E4 DTI splits to the CS-LSDA-DTI schema.')
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--scenarios', nargs='+', default=['random', 'e2', 'e3', 'e4'])
    parser.add_argument('--folds', nargs='+', default=None)
    args = parser.parse_args()
    convert_dataset(input_root=Path(args.input_root), output_root=Path(args.output_root), scenarios=args.scenarios, folds=args.folds)
import argparse
from pathlib import Path
import pandas as pd
COLUMNS = ['drug_id', 'target_id', 'smiles', 'protein_sequence', 'label']

def read_biosnap(path: str | Path) -> pd.DataFrame:
    """Read BioSNAP-style whitespace separated rows without a header."""
    df = pd.read_csv(path, sep='\\s+', names=COLUMNS, engine='python')
    df['label'] = df['label'].astype(float)
    return df

def split_random(df: pd.DataFrame, train_ratio: float=0.7, valid_ratio: float=0.1, test_ratio: float=0.2, seed: int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-06:
        raise ValueError(f'Split ratios must sum to 1.0, got {total}')
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    valid = shuffled.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_valid:].reset_index(drop=True)
    return (train, valid, test)

def split_random_cli() -> None:
    parser = argparse.ArgumentParser(description='Create random 7:1:2 interaction-pair splits.')
    parser.add_argument('--input', default='BIOSNAP.txt', help='Input BioSNAP txt file.')
    parser.add_argument('--out_dir', default='data/biosnap_random_7_1_2', help='Output split directory.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_biosnap(args.input)
    train, valid, test = split_random(df, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio, seed=args.seed)
    train.to_csv(out_dir / 'train.csv', index=False)
    valid.to_csv(out_dir / 'valid.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)
    print(f'Saved {len(train)} train, {len(valid)} valid, {len(test)} test rows to {out_dir}')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
Split = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

def split_entities(values: pd.Series, seed: int, train_ratio: float=0.7, valid_ratio: float=0.1) -> tuple[set[str], set[str], set[str]]:
    unique = np.array(sorted(values.astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = set(unique[:n_train])
    valid = set(unique[n_train:n_train + n_valid])
    test = set(unique[n_train + n_valid:])
    return (train, valid, test)

def make_balanced_entity_folds(values: pd.Series, folds: int, seed: int) -> list[set[str]]:
    """Create disjoint entity folds balanced by their interaction counts."""
    counts = values.astype(str).value_counts()
    rng = np.random.default_rng(seed)
    entities = np.array(counts.index.to_list(), dtype=object)
    rng.shuffle(entities)
    entities = sorted(entities, key=lambda x: int(counts[x]), reverse=True)
    fold_sets = [set() for _ in range(folds)]
    fold_weights = np.zeros(folds, dtype=np.int64)
    for entity in entities:
        fold = int(np.argmin(fold_weights))
        fold_sets[fold].add(str(entity))
        fold_weights[fold] += int(counts[entity])
    return fold_sets

def make_bipartite_entity_parts(df: pd.DataFrame, parts: int, seed: int, iterations: int=30, balance_penalty: float=200.0) -> tuple[list[set[str]], list[set[str]]]:
    """Co-partition drugs and targets for double-cold binding splits."""
    drug = df['drug_id'].astype(str)
    target = df['target_id'].astype(str)
    drug_counts = drug.value_counts().to_dict()
    target_counts = target.value_counts().to_dict()
    drug_to_targets: dict[str, list[str]] = {}
    target_to_drugs: dict[str, list[str]] = {}
    for d, t in zip(drug, target, strict=True):
        drug_to_targets.setdefault(d, []).append(t)
        target_to_drugs.setdefault(t, []).append(d)
    rng = np.random.default_rng(seed)

    def initialize(items: list[str], counts: dict[str, int]) -> dict[str, int]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: counts[x], reverse=True)
        loads = np.zeros(parts, dtype=np.int64)
        labels: dict[str, int] = {}
        for item in shuffled:
            part = int(np.argmin(loads))
            labels[item] = part
            loads[part] += counts[item]
        return labels

    def assign_by_neighbors(items: list[str], counts: dict[str, int], neighbors: dict[str, list[str]], other_labels: dict[str, int]) -> dict[str, int]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: counts[x], reverse=True)
        loads = np.zeros(parts, dtype=np.int64)
        capacity = max(1.0, sum(counts.values()) / parts)
        labels: dict[str, int] = {}
        for item in shuffled:
            neighbor_counts = np.zeros(parts, dtype=np.float64)
            for neighbor in neighbors.get(item, []):
                neighbor_counts[other_labels[neighbor]] += 1.0
            overflow = np.maximum(0.0, loads + counts[item] - capacity) / capacity
            scores = neighbor_counts - balance_penalty * overflow
            part = int(np.argmax(scores))
            labels[item] = part
            loads[part] += counts[item]
        return labels
    drug_labels = initialize(list(drug_counts), drug_counts)
    target_labels = initialize(list(target_counts), target_counts)
    for _ in range(iterations):
        target_labels = assign_by_neighbors(list(target_counts), target_counts, target_to_drugs, drug_labels)
        drug_labels = assign_by_neighbors(list(drug_counts), drug_counts, drug_to_targets, target_labels)
    drug_parts = [set() for _ in range(parts)]
    target_parts = [set() for _ in range(parts)]
    for d, part in drug_labels.items():
        drug_parts[part].add(d)
    for t, part in target_labels.items():
        target_parts[part].add(t)
    return (drug_parts, target_parts)

def choose_entities_for_budget(candidate_entities: set[str], entity_rows: pd.Series, entity_positives: pd.Series, target_rows: int, target_positives: int, seed: int) -> set[str]:
    """Choose entities close to both row-count and positive-count budgets."""
    if target_rows <= 0 or not candidate_entities:
        return set()
    rng = np.random.default_rng(seed)
    remaining = list(candidate_entities)
    rng.shuffle(remaining)
    chosen: set[str] = set()
    current_rows = 0
    current_positives = 0

    def objective(rows: int, positives: int) -> float:
        row_error = abs(rows - target_rows) / max(1, target_rows)
        pos_error = abs(positives - target_positives) / max(1, target_positives)
        return row_error + pos_error
    while remaining:
        current_obj = objective(current_rows, current_positives)
        best_idx = -1
        best_obj = float('inf')
        for idx, entity in enumerate(remaining):
            rows = current_rows + int(entity_rows.get(entity, 0))
            positives = current_positives + int(entity_positives.get(entity, 0))
            obj = objective(rows, positives)
            if obj < best_obj:
                best_idx = idx
                best_obj = obj
        if best_idx < 0 or (chosen and best_obj > current_obj and (current_rows >= target_rows * 0.95)):
            break
        entity = remaining.pop(best_idx)
        chosen.add(entity)
        current_rows += int(entity_rows.get(entity, 0))
        current_positives += int(entity_positives.get(entity, 0))
        if current_rows >= target_rows and current_positives >= target_positives:
            if objective(current_rows, current_positives) <= current_obj or current_rows <= target_rows * 1.05:
                break
    return chosen

def write_split(out_dir: Path, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / 'train.csv', index=False)
    valid.to_csv(out_dir / 'valid.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)

def summarize(df: pd.DataFrame) -> dict[str, int | float]:
    return {'rows': int(len(df)), 'positive': int((df['label'].astype(float) == 1.0).sum()), 'negative': int((df['label'].astype(float) == 0.0).sum()), 'drugs': int(df['drug_id'].astype(str).nunique()), 'targets': int(df['target_id'].astype(str).nunique()), 'positive_rate': float((df['label'].astype(float) == 1.0).mean()) if len(df) else 0.0}

def shuffle_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, seed: int) -> Split:
    return (train.sample(frac=1.0, random_state=seed).reset_index(drop=True), valid.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True), test.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True))

def make_repeated_holdout_fold(df: pd.DataFrame, scenario: str, seed: int) -> Split:
    drug = df['drug_id'].astype(str)
    target = df['target_id'].astype(str)
    if scenario == 'unseen_drug':
        train_drugs, valid_drugs, test_drugs = split_entities(drug, seed)
        train = df[drug.isin(train_drugs)]
        valid = df[drug.isin(valid_drugs)]
        test = df[drug.isin(test_drugs)]
    elif scenario == 'unseen_target':
        train_targets, valid_targets, test_targets = split_entities(target, seed)
        train = df[target.isin(train_targets)]
        valid = df[target.isin(valid_targets)]
        test = df[target.isin(test_targets)]
    elif scenario == 'unseen_binding':
        train_drugs, valid_drugs, test_drugs = split_entities(drug, seed)
        train_targets, valid_targets, test_targets = split_entities(target, seed + 100000)
        train = df[drug.isin(train_drugs) & target.isin(train_targets)]
        valid = df[drug.isin(valid_drugs) & target.isin(valid_targets)]
        test = df[drug.isin(test_drugs) & target.isin(test_targets)]
    else:
        raise ValueError(f'Unsupported scenario: {scenario}')
    return shuffle_split(train, valid, test, seed)

def make_tridti_fold(df: pd.DataFrame, scenario: str, fold: int, folds: int, seed: int, drug_folds: list[set[str]], target_folds: list[set[str]], binding_drug_parts: list[set[str]], binding_target_parts: list[set[str]]) -> Split:
    """TriDTI-style five-fold cold split with a 7:1:2 train/valid/test target."""
    drug = df['drug_id'].astype(str)
    target = df['target_id'].astype(str)
    target_valid_rows = int(round(len(df) * 0.1))
    target_valid_positives = int(round((df['label'].astype(float) == 1.0).sum() * 0.1))
    if scenario == 'unseen_drug':
        test_drugs = drug_folds[fold]
        remaining_drugs = set(drug.unique()).difference(test_drugs)
        drug_rows = drug.value_counts()
        drug_positives = df[df['label'].astype(float) == 1.0]['drug_id'].astype(str).value_counts()
        valid_drugs = choose_entities_for_budget(remaining_drugs, drug_rows, drug_positives, target_valid_rows, target_valid_positives, seed + 10000 + fold)
        train_drugs = remaining_drugs.difference(valid_drugs)
        train = df[drug.isin(train_drugs)]
        valid = df[drug.isin(valid_drugs)]
        test = df[drug.isin(test_drugs)]
    elif scenario == 'unseen_target':
        test_targets = target_folds[fold]
        remaining_targets = set(target.unique()).difference(test_targets)
        target_rows = target.value_counts()
        target_positives = df[df['label'].astype(float) == 1.0]['target_id'].astype(str).value_counts()
        valid_targets = choose_entities_for_budget(remaining_targets, target_rows, target_positives, target_valid_rows, target_valid_positives, seed + 20000 + fold)
        train_targets = remaining_targets.difference(valid_targets)
        train = df[target.isin(train_targets)]
        valid = df[target.isin(valid_targets)]
        test = df[target.isin(test_targets)]
    elif scenario == 'unseen_binding':
        parts = len(binding_drug_parts)
        test_part_ids = {2 * fold % parts, (2 * fold + 1) % parts}
        valid_part_ids = {(2 * fold + 2) % parts}
        train_part_ids = set(range(parts)).difference(test_part_ids | valid_part_ids)
        drug_part = {entity: part for part, entities in enumerate(binding_drug_parts) for entity in entities}
        target_part = {entity: part for part, entities in enumerate(binding_target_parts) for entity in entities}
        dpart = drug.map(drug_part)
        tpart = target.map(target_part)
        same_block = dpart == tpart
        train = df[same_block & dpart.isin(train_part_ids)]
        valid = df[same_block & dpart.isin(valid_part_ids)]
        test = df[same_block & dpart.isin(test_part_ids)]
    else:
        raise ValueError(f'Unsupported scenario: {scenario}')
    return shuffle_split(train, valid, test, seed + fold)

def summarize_leakage(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    train_drugs = set(train['drug_id'].astype(str))
    valid_drugs = set(valid['drug_id'].astype(str))
    test_drugs = set(test['drug_id'].astype(str))
    train_targets = set(train['target_id'].astype(str))
    valid_targets = set(valid['target_id'].astype(str))
    test_targets = set(test['target_id'].astype(str))
    return {'drug_train_valid_overlap': len(train_drugs & valid_drugs), 'drug_train_test_overlap': len(train_drugs & test_drugs), 'drug_valid_test_overlap': len(valid_drugs & test_drugs), 'target_train_valid_overlap': len(train_targets & valid_targets), 'target_train_test_overlap': len(train_targets & test_targets), 'target_valid_test_overlap': len(valid_targets & test_targets)}

def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        missing = set(COLUMNS).difference(df.columns)
        if missing:
            raise ValueError(f'{path} missing required columns: {sorted(missing)}')
        df = df[COLUMNS].copy()
        df['label'] = df['label'].astype(float)
        return df
    return read_biosnap(path)

def split_cold_cli() -> None:
    parser = argparse.ArgumentParser(description='Create TriDTI-style 5-fold cold-start BioSNAP splits.')
    parser.add_argument('--input', default='BIOSNAP.txt')
    parser.add_argument('--out_dir', default='data/biosnap_cold_cv')
    parser.add_argument('--scenarios', nargs='+', default=['unseen_drug', 'unseen_target', 'unseen_binding'], choices=['unseen_drug', 'unseen_target', 'unseen_binding'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', choices=['tridti', 'repeated_holdout'], default='tridti', help='tridti creates disjoint test folds and targets a 7:1:2 train/valid/test split; repeated_holdout keeps the previous seed-per-fold behavior.')
    args = parser.parse_args()
    df = read_input(Path(args.input))
    out_root = Path(args.out_dir)
    summary: dict[str, object] = {'strategy': args.strategy, 'folds': args.folds, 'seed': args.seed, 'input': str(args.input), 'source': summarize(df), 'scenarios': {}}
    drug_folds = make_balanced_entity_folds(df['drug_id'], args.folds, args.seed)
    target_folds = make_balanced_entity_folds(df['target_id'], args.folds, args.seed + 100000)
    binding_drug_parts, binding_target_parts = make_bipartite_entity_parts(df, parts=args.folds * 2, seed=args.seed + 200000)
    for scenario in args.scenarios:
        scenario_summary = {}
        for fold in range(args.folds):
            fold_seed = args.seed + fold
            if args.strategy == 'tridti':
                train, valid, test = make_tridti_fold(df=df, scenario=scenario, fold=fold, folds=args.folds, seed=args.seed, drug_folds=drug_folds, target_folds=target_folds, binding_drug_parts=binding_drug_parts, binding_target_parts=binding_target_parts)
            else:
                train, valid, test = make_repeated_holdout_fold(df, scenario, fold_seed)
            fold_dir = out_root / scenario / f'fold_{fold + 1}'
            write_split(fold_dir, train, valid, test)
            kept_rows = len(train) + len(valid) + len(test)
            scenario_summary[f'fold_{fold + 1}'] = {'train': summarize(train), 'valid': summarize(valid), 'test': summarize(test), 'kept_rows': int(kept_rows), 'dropped_rows': int(len(df) - kept_rows), 'row_ratios': {'train': float(len(train) / kept_rows) if kept_rows else 0.0, 'valid': float(len(valid) / kept_rows) if kept_rows else 0.0, 'test': float(len(test) / kept_rows) if kept_rows else 0.0}, 'entity_overlap': summarize_leakage(train, valid, test)}
            print(f'{scenario} fold {fold + 1}: train={len(train)}, valid={len(valid)}, test={len(test)}, dropped={len(df) - kept_rows} -> {fold_dir}')
        summary['scenarios'][scenario] = scenario_summary
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    import sys
    commands = {
        "convert": convert_cli,
        "split-random": split_random_cli,
        "split-cold": split_cold_cli,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python data.py {convert|split-random|split-cold} [arguments]")
    command = commands[sys.argv.pop(1)]
    command()
