from __future__ import annotations

import inspect
from pathlib import Path

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
from torch import nn
import torch.nn.functional as F

class TokenCNNEncoder(nn.Module):
    """Trainable token embedding followed by length-preserving 1D convolutions."""

    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int, kernels: list[int], dropout: float, padding_index: int=0) -> None:
        super().__init__()
        if not kernels or any((kernel <= 0 or kernel % 2 == 0 for kernel in kernels)):
            raise ValueError('TokenCNNEncoder kernels must be non-empty positive odd integers')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        channels = [embedding_dim] + [output_dim] * len(kernels)
        self.convs = nn.ModuleList([nn.Conv1d(channels[index], channels[index + 1], kernel_size=kernel, padding=kernel // 2) for index, kernel in enumerate(kernels)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor | None=None) -> torch.Tensor:
        features = self.embedding(token_ids).transpose(1, 2)
        for convolution in self.convs:
            features = self.dropout(F.gelu(convolution(features)))
            if token_mask is not None:
                features = features.masked_fill(token_mask.unsqueeze(1), 0.0)
        return features.transpose(1, 2)
import torch
from torch import nn

class MultiScaleAdapter(nn.Module):

    def __init__(self, hidden_dim: int, kernels: list[int], dropout: float=0.1) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel, padding=kernel // 2) for kernel in kernels])
        self.proj = nn.Linear(hidden_dim * len(kernels), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        feats = [conv(x_t).transpose(1, 2) for conv in self.convs]
        out = self.proj(torch.cat(feats, dim=-1))
        out = self.dropout(out)
        return self.norm(out + residual)

class ACLPatch(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int=3, gamma_init: float=0.0001) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Parameter(torch.full((hidden_dim,), gamma_init))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        branch = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        branch = self.norm(branch)
        if key_padding_mask is not None:
            branch = branch.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x + self.gamma * branch
import torch
from torch import nn

class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float=0.1, attention_dropout: float=0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, hidden_dim), nn.Dropout(dropout))

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, query_mask: torch.Tensor | None=None, key_value_mask: torch.Tensor | None=None) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_value_mask, need_weights=False)
        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        return x
import torch
from torch import nn

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.last_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.last_weights = weights.detach()
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)

class MaskedMeanPooling(nn.Module):
    """Mean pooling over valid tokens with the same interface as attention pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.last_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        if mask is None:
            length = max(tokens.size(1), 1)
            weights = tokens.new_full(tokens.shape[:2], 1.0 / length)
        else:
            weights = (~mask).to(dtype=tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.last_weights = weights.detach()
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
import torch
from torch import nn
import torch.nn.functional as F

class DenseGINE(nn.Module):
    """Small edge-aware GIN encoder without a torch-geometric dependency."""

    def __init__(self, node_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(5, hidden_dim)
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor, edge_features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(node_features)
        edge = self.edge_proj(edge_features)
        for mlp, norm in zip(self.mlps, self.norms):
            messages = F.gelu(hidden.unsqueeze(1) + edge)
            messages = messages * adjacency.unsqueeze(-1)
            aggregated = messages.sum(dim=2)
            hidden = norm(hidden + mlp(hidden + aggregated))
            hidden = hidden.masked_fill(node_mask.unsqueeze(-1), 0.0)
        valid = (~node_mask).unsqueeze(-1).float()
        return hidden.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

class ProteinSequenceCNN(nn.Module):

    def __init__(self, hidden_dim: int, kernels: list[int], dropout: float) -> None:
        super().__init__()
        branch_dim = hidden_dim // len(kernels)
        self.embedding = nn.Embedding(21, branch_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(branch_dim, branch_dim, kernel_size=kernel, padding=kernel // 2) for kernel in kernels])
        self.output = nn.Sequential(nn.Linear(branch_dim * len(kernels), hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence).transpose(1, 2)
        valid = (~sequence_mask).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            features = F.gelu(conv(embedded)).masked_fill(~valid, -torch.inf)
            pooled.append(features.max(dim=-1).values)
        return self.output(torch.cat(pooled, dim=-1))

class ResidualGate(nn.Module):

    def __init__(self, hidden_dim: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, base: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([base, auxiliary], dim=-1)))
        return base + self.alpha * gate * auxiliary

class InteractionAwareResidualGate(nn.Module):

    def __init__(self, hidden_dim: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.gate = nn.Sequential(nn.Linear(5 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, interaction: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        context = torch.cat([drug, auxiliary, interaction, torch.abs(drug - protein), drug * protein], dim=-1)
        gate = torch.sigmoid(self.gate(context))
        return drug + self.alpha * gate * auxiliary
import math
import torch
from torch import nn
import torch.nn.functional as F

def _masked_mean_pool_windows(tokens: torch.Tensor, mask: torch.Tensor | None, window_size: int, token_scores: torch.Tensor | None=None, score_scale: float=1.0) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, length, dim = tokens.shape
    n_windows = math.ceil(length / window_size)
    pad_len = n_windows * window_size - length
    if pad_len:
        tokens = F.pad(tokens, (0, 0, 0, pad_len))
        if token_scores is not None:
            token_scores = F.pad(token_scores, (0, pad_len), value=0.0)
        if mask is not None:
            mask = F.pad(mask, (0, pad_len), value=True)
    tokens = tokens.view(bsz, n_windows, window_size, dim)
    if token_scores is not None:
        token_scores = token_scores.view(bsz, n_windows, window_size)
    if mask is None:
        if token_scores is None:
            pooled = tokens.mean(dim=2)
        else:
            weights = (1.0 + score_scale * token_scores).unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-06)
        pooled_mask = None
    else:
        window_mask = mask.view(bsz, n_windows, window_size)
        valid = (~window_mask).unsqueeze(-1)
        if token_scores is None:
            weights = valid.float()
        else:
            weights = (1.0 + score_scale * token_scores).unsqueeze(-1) * valid.float()
        denom = weights.sum(dim=2).clamp_min(1e-06)
        pooled = (tokens * weights).sum(dim=2) / denom
        pooled_mask = window_mask.all(dim=2)
        pooled = pooled.masked_fill(pooled_mask.unsqueeze(-1), 0.0)
    return (pooled, pooled_mask)

def _select_confident_subset(tokens: torch.Tensor, mask: torch.Tensor | None, token_scores: torch.Tensor | None, stride: int, topk_ratio: float=0.5) -> tuple[torch.Tensor, torch.Tensor | None]:
    stride_subset = tokens[:, ::stride, :]
    stride_mask = mask[:, ::stride] if mask is not None else None
    if token_scores is None:
        return (stride_subset, stride_mask)
    bsz, length, dim = tokens.shape
    k = max(1, math.ceil(length / stride * topk_ratio))
    scores = token_scores
    if mask is not None:
        scores = scores.masked_fill(mask, -torch.inf)
    top_indices = torch.topk(scores, k=min(k, length), dim=1).indices
    top_indices = top_indices.sort(dim=1).values
    gather_index = top_indices.unsqueeze(-1).expand(-1, -1, dim)
    top_subset = torch.gather(tokens, dim=1, index=gather_index)
    if mask is None:
        top_mask = None
    else:
        top_mask = torch.gather(mask, dim=1, index=top_indices)
    subset = torch.cat([stride_subset, top_subset], dim=1)
    if stride_mask is None or top_mask is None:
        subset_mask = None
    else:
        subset_mask = torch.cat([stride_mask, top_mask], dim=1)
    return (subset, subset_mask)

class SDAFusionBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, window_size: int, dropout: float=0.1, attention_dropout: float=0.1, acl_kernel_size: int=3, acl_gamma_init: float=0.0001, use_acl: bool=True, use_reverse_cross_attention: bool=True, score_scale: float=1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.score_scale = score_scale
        self.use_reverse_cross_attention = use_reverse_cross_attention
        self.drug_to_protein = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.protein_to_drug = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.drug_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None
        self.protein_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, drug_mask: torch.Tensor | None=None, protein_mask: torch.Tensor | None=None, protein_scores: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        protein_windows, protein_window_mask = _masked_mean_pool_windows(protein, protein_mask, self.window_size, protein_scores, self.score_scale)
        drug = self.drug_to_protein(drug, protein_windows, drug_mask, protein_window_mask)
        if self.use_reverse_cross_attention:
            protein = self.protein_to_drug(protein, drug, protein_mask, drug_mask)
        if self.drug_acl is not None:
            drug = self.drug_acl(drug, drug_mask)
        if self.protein_acl is not None:
            protein = self.protein_acl(protein, protein_mask)
        return (drug, protein)

class LDAFusionBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, stride: int, dropout: float=0.1, attention_dropout: float=0.1, acl_kernel_size: int=3, acl_gamma_init: float=0.0001, use_acl: bool=True, use_reverse_cross_attention: bool=True, topk_ratio: float=0.5) -> None:
        super().__init__()
        self.stride = stride
        self.topk_ratio = topk_ratio
        self.use_reverse_cross_attention = use_reverse_cross_attention
        self.drug_to_protein = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.protein_to_drug = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.drug_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None
        self.protein_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, drug_mask: torch.Tensor | None=None, protein_mask: torch.Tensor | None=None, protein_scores: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        protein_subset, protein_subset_mask = _select_confident_subset(protein, protein_mask, protein_scores, self.stride, self.topk_ratio)
        drug = self.drug_to_protein(drug, protein_subset, drug_mask, protein_subset_mask)
        if self.use_reverse_cross_attention:
            protein = self.protein_to_drug(protein, drug, protein_mask, drug_mask)
        if self.drug_acl is not None:
            drug = self.drug_acl(drug, drug_mask)
        if self.protein_acl is not None:
            protein = self.protein_acl(protein, protein_mask)
        return (drug, protein)

class GlobalCrossAttentionFusionBlock(nn.Module):
    """Bidirectional full cross-attention baseline without LSDA compression."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float=0.1, attention_dropout: float=0.1, acl_kernel_size: int=3, acl_gamma_init: float=0.0001, use_acl: bool=True, use_reverse_cross_attention: bool=True) -> None:
        super().__init__()
        self.use_reverse_cross_attention = use_reverse_cross_attention
        self.drug_to_protein = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.protein_to_drug = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.drug_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None
        self.protein_acl = ACLPatch(hidden_dim, acl_kernel_size, acl_gamma_init) if use_acl else None

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, drug_mask: torch.Tensor | None=None, protein_mask: torch.Tensor | None=None, protein_scores: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        del protein_scores
        drug = self.drug_to_protein(drug, protein, drug_mask, protein_mask)
        if self.use_reverse_cross_attention:
            protein = self.protein_to_drug(protein, drug, protein_mask, drug_mask)
        if self.drug_acl is not None:
            drug = self.drug_acl(drug, drug_mask)
        if self.protein_acl is not None:
            protein = self.protein_acl(protein, protein_mask)
        return (drug, protein)

class LSDAFusion(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, lsda_types: list[str], sda_window_sizes: list[int], lda_strides: list[int], dropout: float=0.1, attention_dropout: float=0.1, acl_kernel_size: int=3, acl_gamma_init: float=0.0001, use_acl: bool=True, use_reverse_cross_attention: bool=True, sda_score_scale: float=1.0, lda_topk_ratio: float=0.5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sda_i = 0
        lda_i = 0
        for layer_type in lsda_types:
            if layer_type == 'sda':
                layers.append(SDAFusionBlock(hidden_dim, num_heads, ffn_dim, sda_window_sizes[sda_i], dropout, attention_dropout, acl_kernel_size, acl_gamma_init, use_acl, use_reverse_cross_attention, sda_score_scale))
                sda_i += 1
            elif layer_type == 'lda':
                layers.append(LDAFusionBlock(hidden_dim, num_heads, ffn_dim, lda_strides[lda_i], dropout, attention_dropout, acl_kernel_size, acl_gamma_init, use_acl, use_reverse_cross_attention, lda_topk_ratio))
                lda_i += 1
            elif layer_type == 'global':
                layers.append(GlobalCrossAttentionFusionBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout, acl_kernel_size, acl_gamma_init, use_acl, use_reverse_cross_attention))
            else:
                raise ValueError(f'Unsupported LSDA layer type: {layer_type}')
        self.layers = nn.ModuleList(layers)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor, drug_mask: torch.Tensor | None=None, protein_mask: torch.Tensor | None=None, protein_scores: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            drug, protein = layer(drug, protein, drug_mask, protein_mask, protein_scores)
        return (drug, protein)
import torch
from torch import nn

class PseudoBindingPrior(nn.Module):
    """Infer residue-level binding priors from frozen protein LM embeddings."""

    def __init__(self, hidden_dim: int, dropout: float=0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, protein: torch.Tensor, protein_mask: torch.Tensor | None=None) -> torch.Tensor:
        scores = torch.sigmoid(self.scorer(protein).squeeze(-1))
        if protein_mask is not None:
            scores = scores.masked_fill(protein_mask, 0.0)
        return scores

class CSLSDADTI(nn.Module):

    def __init__(self, drug_input_dim: int, protein_input_dim: int, hidden_dim: int=512, num_heads: int=8, ffn_dim: int=2048, dropout: float=0.1, attention_dropout: float=0.1, drug_kernels: list[int] | None=None, protein_kernels: list[int] | None=None, lsda_types: list[str] | None=None, sda_window_sizes: list[int] | None=None, lda_strides: list[int] | None=None, acl_kernel_size: int=3, acl_gamma_init: float=0.0001, use_multiscale_adapter: bool=True, use_acl: bool=True, use_reverse_cross_attention: bool=True, pooling_type: str='attention', use_pairwise_matching_features: bool=True, fusion_mode: str='lsda', drug_encoder_type: str='plm', protein_encoder_type: str='plm', raw_token_embedding_dim: int=128, raw_cnn_kernels: list[int] | None=None, use_pseudo_binding_prior: bool=True, prior_modulation_scale: float=0.5, sda_score_scale: float=1.0, lda_topk_ratio: float=0.5, use_multiview_residual: bool=False, graph_hidden_dim: int=128, graph_layers: int=3, protein_sequence_kernels: list[int] | None=None, multiview_alpha: float=0.2, use_graph_residual: bool=True, use_sequence_residual: bool=True, use_interaction_aware_graph_gate: bool=False) -> None:
        super().__init__()
        drug_kernels = drug_kernels or [1, 3, 5]
        protein_kernels = protein_kernels or [1, 5, 9]
        lsda_types = lsda_types or ['sda', 'lda', 'sda', 'lda']
        sda_window_sizes = sda_window_sizes or [64, 128]
        lda_strides = lda_strides or [8, 16]
        raw_cnn_kernels = raw_cnn_kernels or [3, 5, 7]
        if drug_encoder_type not in {'plm', 'smiles_cnn'}:
            raise ValueError(f'Unsupported drug_encoder_type: {drug_encoder_type}')
        if protein_encoder_type not in {'plm', 'aa_cnn'}:
            raise ValueError(f'Unsupported protein_encoder_type: {protein_encoder_type}')
        self.drug_encoder_type = drug_encoder_type
        self.protein_encoder_type = protein_encoder_type
        self.raw_drug_encoder = TokenCNNEncoder(SMILES_VOCAB_SIZE, raw_token_embedding_dim, drug_input_dim, raw_cnn_kernels, dropout) if drug_encoder_type == 'smiles_cnn' else None
        self.raw_protein_encoder = TokenCNNEncoder(AA_VOCAB_SIZE, raw_token_embedding_dim, protein_input_dim, raw_cnn_kernels, dropout) if protein_encoder_type == 'aa_cnn' else None
        self.drug_proj = nn.Linear(drug_input_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_input_dim, hidden_dim)
        self.drug_adapter = MultiScaleAdapter(hidden_dim, drug_kernels, dropout) if use_multiscale_adapter else nn.Identity()
        self.protein_adapter = MultiScaleAdapter(hidden_dim, protein_kernels, dropout) if use_multiscale_adapter else nn.Identity()
        self.use_pseudo_binding_prior = use_pseudo_binding_prior
        self.use_pairwise_matching_features = use_pairwise_matching_features
        self.use_multiview_residual = use_multiview_residual
        self.use_graph_residual = use_multiview_residual and use_graph_residual
        self.use_sequence_residual = use_multiview_residual and use_sequence_residual
        self.use_interaction_aware_graph_gate = self.use_graph_residual and use_interaction_aware_graph_gate
        self.prior_modulation_scale = prior_modulation_scale
        self.binding_prior = PseudoBindingPrior(hidden_dim, dropout) if use_pseudo_binding_prior else None
        if fusion_mode not in {'lsda', 'concat'}:
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')
        self.fusion_mode = fusion_mode
        self.fusion = LSDAFusion(hidden_dim=hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim, lsda_types=lsda_types, sda_window_sizes=sda_window_sizes, lda_strides=lda_strides, dropout=dropout, attention_dropout=attention_dropout, acl_kernel_size=acl_kernel_size, acl_gamma_init=acl_gamma_init, use_acl=use_acl, use_reverse_cross_attention=use_reverse_cross_attention, sda_score_scale=sda_score_scale, lda_topk_ratio=lda_topk_ratio) if fusion_mode == 'lsda' else None
        if pooling_type == 'attention':
            pooling_factory = lambda: AttentionPooling(hidden_dim)
        elif pooling_type == 'mean':
            pooling_factory = MaskedMeanPooling
        else:
            raise ValueError(f'Unsupported pooling_type: {pooling_type}')
        self.drug_pool = pooling_factory()
        self.protein_pool = pooling_factory()
        self.interaction_pool = pooling_factory()
        if self.use_graph_residual:
            self.graph_encoder = DenseGINE(128, graph_hidden_dim, graph_layers, dropout)
            self.graph_proj = nn.Linear(graph_hidden_dim, hidden_dim)
            if self.use_interaction_aware_graph_gate:
                self.drug_residual_gate = InteractionAwareResidualGate(hidden_dim, multiview_alpha, dropout)
            else:
                self.drug_residual_gate = ResidualGate(hidden_dim, multiview_alpha, dropout)
        if self.use_sequence_residual:
            self.protein_sequence_encoder = ProteinSequenceCNN(hidden_dim, protein_sequence_kernels or [3, 5, 7], dropout)
            self.protein_residual_gate = ResidualGate(hidden_dim, multiview_alpha, dropout)
        head_input_dim = (5 if use_pairwise_matching_features else 3) * hidden_dim
        self.head = nn.Sequential(nn.Linear(head_input_dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, drug_embedding: torch.Tensor, protein_embedding: torch.Tensor, drug_mask: torch.Tensor | None=None, protein_mask: torch.Tensor | None=None, prior_scale: float | None=None, graph_node_features: torch.Tensor | None=None, graph_adjacency: torch.Tensor | None=None, graph_edge_features: torch.Tensor | None=None, graph_node_mask: torch.Tensor | None=None, protein_sequence: torch.Tensor | None=None, protein_sequence_mask: torch.Tensor | None=None, raw_smiles_tokens: torch.Tensor | None=None, raw_smiles_mask: torch.Tensor | None=None, raw_protein_tokens: torch.Tensor | None=None, raw_protein_mask: torch.Tensor | None=None, return_features: bool=False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.raw_drug_encoder is not None:
            if raw_smiles_tokens is None or raw_smiles_mask is None:
                raise ValueError('SMILES tokens are required for drug_encoder_type=smiles_cnn')
            drug_embedding = self.raw_drug_encoder(raw_smiles_tokens, raw_smiles_mask)
            drug_mask = raw_smiles_mask
        if self.raw_protein_encoder is not None:
            if raw_protein_tokens is None or raw_protein_mask is None:
                raise ValueError('Protein tokens are required for protein_encoder_type=aa_cnn')
            protein_embedding = self.raw_protein_encoder(raw_protein_tokens, raw_protein_mask)
            protein_mask = raw_protein_mask
        drug = self.drug_adapter(self.drug_proj(drug_embedding))
        protein = self.protein_adapter(self.protein_proj(protein_embedding))
        if drug_mask is not None:
            drug = drug.masked_fill(drug_mask.unsqueeze(-1), 0.0)
        if protein_mask is not None:
            protein = protein.masked_fill(protein_mask.unsqueeze(-1), 0.0)
        protein_scores = None
        if self.binding_prior is not None:
            protein_scores = self.binding_prior(protein, protein_mask)
            scale = self.prior_modulation_scale if prior_scale is None else prior_scale
            protein = protein * (1.0 + scale * protein_scores.unsqueeze(-1))
            if protein_mask is not None:
                protein = protein.masked_fill(protein_mask.unsqueeze(-1), 0.0)
        if self.fusion is not None:
            drug, protein = self.fusion(drug, protein, drug_mask, protein_mask, protein_scores)
        drug_pool = self.drug_pool(drug, drug_mask)
        protein_pool = self.protein_pool(protein, protein_mask)
        interaction_tokens = torch.cat([drug, protein], dim=1)
        interaction_mask = None
        if drug_mask is not None and protein_mask is not None:
            interaction_mask = torch.cat([drug_mask, protein_mask], dim=1)
        interaction_pool = self.interaction_pool(interaction_tokens, interaction_mask)
        if self.use_graph_residual:
            required_graph = [graph_node_features, graph_adjacency, graph_edge_features, graph_node_mask]
            if any((value is None for value in required_graph)):
                raise ValueError('Graph residual features are required by the configured model.')
            graph_feature = self.graph_proj(self.graph_encoder(graph_node_features, graph_adjacency, graph_edge_features, graph_node_mask))
            if self.use_interaction_aware_graph_gate:
                drug_pool = self.drug_residual_gate(drug_pool, protein_pool, interaction_pool, graph_feature)
            else:
                drug_pool = self.drug_residual_gate(drug_pool, graph_feature)
        if self.use_sequence_residual:
            if protein_sequence is None or protein_sequence_mask is None:
                raise ValueError('Protein sequence residual features are required by the configured model.')
            protein_sequence_feature = self.protein_sequence_encoder(protein_sequence, protein_sequence_mask)
            protein_pool = self.protein_residual_gate(protein_pool, protein_sequence_feature)
        head_features = [drug_pool, protein_pool, interaction_pool]
        if self.use_pairwise_matching_features:
            head_features.extend([torch.abs(drug_pool - protein_pool), drug_pool * protein_pool])
        h = torch.cat(head_features, dim=-1)
        logits = self.head(h).squeeze(-1)
        if return_features:
            return (logits, {'drug_pool': drug_pool, 'protein_pool': protein_pool, 'interaction_pool': interaction_pool, 'drug_pool_weights': self.drug_pool.last_weights, 'protein_pool_weights': self.protein_pool.last_weights, 'interaction_pool_weights': self.interaction_pool.last_weights, 'protein_prior_scores': protein_scores})
        return logits

def infer_embedding_dim(embedding_dir: str | Path) -> int:
    directory = Path(embedding_dir)
    first = next(directory.glob("*.pt"), None)
    if first is None:
        raise FileNotFoundError(
            f"No .pt embeddings found in {directory}. Run feature extraction first."
        )
    return int(torch.load(first, map_location="cpu").shape[-1])


def build_model(cfg: dict, device: torch.device, state_dict=None) -> torch.nn.Module:
    model_cfg = cfg["model"]
    if model_cfg.get("architecture", "current") != "current":
        raise ValueError("This compact release supports only the paper's current architecture.")
    drug_dim = infer_embedding_dim(cfg["embedding"]["drug_embedding_dir"])
    protein_dim = infer_embedding_dim(cfg["embedding"]["protein_embedding_dir"])
    kwargs = dict(model_cfg)
    kwargs.pop("architecture", None)
    kwargs.pop("fusion_layers", None)
    kwargs.pop("drug_conditioned_prior", None)
    kwargs.pop("use_fingerprint_branch", None)
    kwargs.pop("freeze_encoder", None)
    supported = inspect.signature(CSLSDADTI).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return CSLSDADTI(drug_input_dim=drug_dim, protein_input_dim=protein_dim, **kwargs).to(device)
