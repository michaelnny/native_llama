"""Llama3 model class"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_seq_len: int = 8192

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

    def __post_init__(self):
        pass

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    caches: Optional[List[Dict[str, torch.Tensor]]] = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads if self.n_kv_heads > 0 else 1
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,  # [batch_size, 1, 1, key_len]
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to (batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Handle KV cache
        if cache and "keys" in cache and "values" in cache:
            keys = torch.cat([cache["keys"], xk], dim=1)
            values = torch.cat([cache["values"], xv], dim=1)
        else:
            keys, values = xk, xv
        # Update cache with current seqlen
        if cache is not None:
            cache = {"keys": keys, "values": values}

        # Repeat KV heads if needed
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention
        queries = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, cachelen + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, cachelen + seqlen, head_dim)

        # scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if attn_mask is not None:
        #     scores = scores + attn_mask
        # scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        # attn_output = torch.matmul(scores, values)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attn_mask,
            is_causal=False,  # We handle causality with explicit mask
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(attn_output)

        return output, cache


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,  # [batch_size, 1, 1, key_len]
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        attn_out, new_cache = self.attention(
            self.attention_norm(x), freqs_cis, attn_mask, cache
        )
        h = x + attn_out

        ff_out = self.feed_forward(self.ffn_norm(h))
        out = h + ff_out
        return out, new_cache


class Transformer(nn.Module):
    """Transformer"""

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers: List[TransformerBlock] = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

        # Placeholder for activation checkpointing function
        self.checkpoint_fn = None

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,  # [batch_size, 1, 1, key_len]
        start_pos: Optional[int] = 0,
        caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
    ) -> ModelOutput:
        """
        Args:
            tokens: input tokens (batch_size, seq_len)
            start_pos: Position to start from (useful for caching)
            attn_mask: Casual attention mask with shape (batch_size, 1, 1, key_len)
            caches: Existing KV cache
            use_cache (bool): Use KV cache
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if attn_mask is not None:
            attn_mask = attn_mask.to(h.device)

        if use_cache and caches is None:
            caches = [{} for _ in range(self.n_layers)]

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = caches[layer_idx] if use_cache else None
            if self.training and self.checkpoint_fn is not None:
                h, new_cache = self.checkpoint_fn(layer, h, freqs_cis, attn_mask)
            else:
                h, new_cache = layer(h, freqs_cis, attn_mask, layer_cache)
            if use_cache:
                caches[layer_idx] = new_cache

        last_hidden = h
        policy_output = self.norm(last_hidden)
        logits = self.output(policy_output)  # [batch_size, seq_len, vocab_size]

        return ModelOutput(logits=logits, caches=caches)

    def set_activation_checkpoint(self, function: Callable) -> None:
        """set activation checkpoint function"""
        self.checkpoint_fn = function

    @staticmethod
    def create_causal_attention_mask(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        batch_size: int,
        device: torch.device,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Creates a boolean causal mask where True means "attend" and False means "mask out".
        This is often used in the `F.scaled_dot_product_attention()`

        Args:
            attention_mask (torch.Tensor): A 2D attention mask of shape (batch_size, key_value_length) or a 4D attention mask of shape (batch_size, 1, query_length, key_value_length).
            sequence_length (int): The sequence length being processed.
            target_length (int): The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            batch_size (torch.Tensor): Batch size.
            device (torch.device): The device to place the 4D attention mask on.
            cache_position (torch.Tensor): Indices depicting the position of the input sequence tokens in the sequence, shape `(sequence_length)`.

        Returns:
            A 4D bool causal mask with shape: (batch_size, 1, sequence_length, target_length)
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.ones(
            sequence_length, target_length, dtype=torch.bool, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.tril(causal_mask)

        if cache_position is not None:
            causal_mask = causal_mask & (
                torch.arange(target_length, device=device)
                <= cache_position.reshape(-1, 1)
            )

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            padding_mask = attention_mask.bool()
            mask_length = attention_mask.shape[-1]
            causal_mask[:, :, :, :mask_length] = (
                causal_mask[:, :, :, :mask_length] & padding_mask[:, None, None, :]
            )

        return causal_mask


def load_pretrained_model(
    ckpt_dir: str,
    max_seq_len: int = 8192,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    quantization: Optional[str] = None,
) -> Transformer:
    """
    handles both regular and DeepSpeed checkpoints.
    """
    import logging
    import json

    logger = logging.getLogger()

    assert ckpt_dir and os.path.exists(ckpt_dir)
    param_file = os.path.join(ckpt_dir, "params.json")
    assert os.path.exists(
        param_file
    ), f"Model config file 'params.json' not found in {ckpt_dir}"

    ckpt_file_00 = os.path.join(ckpt_dir, "consolidated.00.pth")
    ckpt_file_merged = os.path.join(ckpt_dir, "consolidated.pth")
    ckpt_file_ds = os.path.join(ckpt_dir, "pytorch_model.bin") 

    if os.path.exists(ckpt_file_00):
        ckpt_file = ckpt_file_00
    elif os.path.exists(ckpt_file_merged):
        ckpt_file = ckpt_file_merged
    elif os.path.exists(ckpt_file_ds):
        ckpt_file = ckpt_file_ds
    else:
        raise FileNotFoundError(
            f"Neither checkpoint file 'consolidated.00.pth' nor 'consolidated.pth' found in {ckpt_dir}"
        )

    # Load model configuration
    logger.info(f"Loading config file: {param_file!r}")

    with open(param_file, "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(**params)
    model_args.max_seq_len = max_seq_len

    # Initialize the model
    model = Transformer(model_args)
    logger.info(f"Loading checkpoint file: {ckpt_file!r}")
    model_state_dict = torch.load(ckpt_file, map_location="cpu")

    if quantization in ["4bit", "8bit"]:
        logger.info(f"Replacing linear layers with {quantization} quantization...")
        replace_linear_layers(model, quantization, compute_dtype=dtype)

    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(dtype=dtype, device=device)
    del model_state_dict

    return model


def replace_linear_layers(
    module: Transformer, quantization: str = "8bit", compute_dtype=torch.float16
):
    """
    Recursively replace all nn.Linear layers with bnb.nn.Linear4bit layers.

    Args:
        module: The PyTorch module to modify
        quantization: The quantization type, '4bit' or '8bit
        compute_dtype: Compute data type

    """
    import bitsandbytes as bnb

    for name, child in module.named_children():
        if "value_head" in name:  # skip newly initialized layer
            continue
        if isinstance(child, nn.Linear):
            if quantization == "8bit":
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=True,
                    threshold=6.0,
                )
            else:  # 4bit
                new_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    compress_statistics=True,
                    quant_type="nf4",  # Using nested float 4 for better accuracy
                )
            setattr(module, name, new_layer)
        else:
            replace_linear_layers(child, quantization, compute_dtype)
