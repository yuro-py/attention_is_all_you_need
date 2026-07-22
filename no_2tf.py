import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(seq_len, d_model, device="cpu"):
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    i = torch.arange(d_model, device=device).unsqueeze(0).float()
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe  # (seq_len, d_model)

def init_mha(d_model, n_heads):
    assert d_model % n_heads == 0
    return nn.ModuleDict({
        "Wq": nn.Linear(d_model, d_model),
        "Wk": nn.Linear(d_model, d_model),
        "Wv": nn.Linear(d_model, d_model),
        "Wo": nn.Linear(d_model, d_model),
    })


def multi_head_attention(p, q, k, v, n_heads, mask=None):
    B, Tq, D = q.shape
    Tk = k.shape[1]
    hd = D // n_heads

    Q = p["Wq"](q).view(B, Tq, n_heads, hd).transpose(1, 2)  # B,H,Tq,hd
    K = p["Wk"](k).view(B, Tk, n_heads, hd).transpose(1, 2)  # B,H,Tk,hd
    V = p["Wv"](v).view(B, Tk, n_heads, hd).transpose(1, 2)  # B,H,Tk,hd

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)  # B,H,Tq,Tk
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)

    out = attn @ V  # B,H,Tq,hd
    out = out.transpose(1, 2).contiguous().view(B, Tq, D)
    return p["Wo"](out)

def init_ffn(d_model, d_ff):
    return nn.ModuleDict({
        "lin1": nn.Linear(d_model, d_ff),
        "lin2": nn.Linear(d_ff, d_model),
    })


def feed_forward(p, x):
    return p["lin2"](F.relu(p["lin1"](x)))

def init_add_norm(d_model):
    return nn.LayerNorm(d_model)


def add_norm(ln, x, sublayer_out):
    return ln(x + sublayer_out)

def init_encoder_layer(d_model, n_heads, d_ff):
    return nn.ModuleDict({
        "mha": init_mha(d_model, n_heads),
        "norm1": init_add_norm(d_model),
        "ffn": init_ffn(d_model, d_ff),
        "norm2": init_add_norm(d_model),
    })


def encoder_layer(p, x, n_heads, mask=None):
    attn_out = multi_head_attention(p["mha"], x, x, x, n_heads, mask)
    x = add_norm(p["norm1"], x, attn_out)
    ffn_out = feed_forward(p["ffn"], x)
    x = add_norm(p["norm2"], x, ffn_out)
    return x


def init_encoder(N, d_model, n_heads, d_ff):
    return nn.ModuleList([init_encoder_layer(d_model, n_heads, d_ff) for _ in range(N)])


def encoder(layers, x, n_heads, mask=None):
    for layer in layers:
        x = encoder_layer(layer, x, n_heads, mask)
    return x

def init_decoder_layer(d_model, n_heads, d_ff):
    return nn.ModuleDict({
        "mmha": init_mha(d_model, n_heads),   # masked self-attention
        "norm1": init_add_norm(d_model),
        "cmha": init_mha(d_model, n_heads),   # cross-attention over encoder output
        "norm2": init_add_norm(d_model),
        "ffn": init_ffn(d_model, d_ff),
        "norm3": init_add_norm(d_model),
    })


def decoder_layer(p, x, enc_out, n_heads, self_mask=None, cross_mask=None):
    self_attn = multi_head_attention(p["mmha"], x, x, x, n_heads, self_mask)
    x = add_norm(p["norm1"], x, self_attn)
    cross_attn = multi_head_attention(p["cmha"], x, enc_out, enc_out, n_heads, cross_mask)
    x = add_norm(p["norm2"], x, cross_attn)
    ffn_out = feed_forward(p["ffn"], x)
    x = add_norm(p["norm3"], x, ffn_out)
    return x


def init_decoder(N, d_model, n_heads, d_ff):
    return nn.ModuleList([init_decoder_layer(d_model, n_heads, d_ff) for _ in range(N)])


def decoder(layers, x, enc_out, n_heads, self_mask=None, cross_mask=None):
    for layer in layers:
        x = decoder_layer(layer, x, enc_out, n_heads, self_mask, cross_mask)
    return x

def init_transformer(src_vocab, tgt_vocab, d_model=512, n_heads=8, d_ff=2048, N=6, max_len=5000):
    params = nn.ModuleDict({
        "src_embed": nn.Embedding(src_vocab, d_model),
        "tgt_embed": nn.Embedding(tgt_vocab, d_model),
        "encoder": init_encoder(N, d_model, n_heads, d_ff),
        "decoder": init_decoder(N, d_model, n_heads, d_ff),
        "out_proj": nn.Linear(d_model, tgt_vocab),
    })
    config = {"d_model": d_model, "n_heads": n_heads, "max_len": max_len}
    return params, config


def transformer_forward(params, config, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    max_len = config["max_len"]
    device = src.device

    pe = positional_encoding(max_len, d_model, device)

    src_x = params["src_embed"](src) * math.sqrt(d_model) + pe[: src.shape[1]]
    tgt_x = params["tgt_embed"](tgt) * math.sqrt(d_model) + pe[: tgt.shape[1]]

    enc_out = encoder(params["encoder"], src_x, n_heads, src_mask)
    dec_out = decoder(params["decoder"], tgt_x, enc_out, n_heads, tgt_mask, cross_mask)

    logits = params["out_proj"](dec_out)
    return F.softmax(logits, dim=-1)  # Output Probabilities

def causal_mask(size, device="cpu"):
    """Lower-triangular mask for masked (decoder) self-attention."""
    return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)


def padding_mask(seq, pad_idx=0):
    """1 where token is real, 0 where it's padding. Shape: B,1,1,T"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

if __name__ == "__main__":
    torch.manual_seed(0)

    src_vocab, tgt_vocab = 1000, 1000
    d_model, n_heads, d_ff, N, max_len = 64, 4, 256, 2, 100

    params, config = init_transformer(src_vocab, tgt_vocab, d_model, n_heads, d_ff, N, max_len)

    B, Tsrc, Ttgt = 2, 10, 8
    src = torch.randint(1, src_vocab, (B, Tsrc))
    tgt = torch.randint(1, tgt_vocab, (B, Ttgt))

    s_mask = padding_mask(src)
    t_mask = causal_mask(Ttgt, src.device) * padding_mask(tgt)

    probs = transformer_forward(params, config, src, tgt, src_mask=s_mask, tgt_mask=t_mask)
    print("Output Probabilities shape:", probs.shape)  # (B, Ttgt, tgt_vocab)
    print("Sums to 1 over vocab:", torch.allclose(probs.sum(-1), torch.ones(B, Ttgt), atol=1e-5))

    n_params = sum(p.numel() for p in params.parameters())
    print("Total trainable parameters:", n_params)
