import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gpt2_rope_weights(model_state_dict_path: str, cfg: HookedTransformerConfig):
    gpt2_rope = torch.load(model_state_dict_path, map_location=torch.device("cpu"))
    gpt2_rope = gpt2_rope["model_state_dict"]

    state_dict = {}

    state_dict["embed.W_E"] = gpt2_rope["transformer.wte.weight"]
    state_dict["pos_embed.W_pos"] = torch.zeros(cfg.n_ctx, cfg.d_model, dtype=cfg.dtype)

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2_rope[f"transformer.h.{l}.ln_1.weight"]
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2_rope[f"transformer.h.{l}.attn.c_attn.weight"]

        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        # print("W_Q shape ", W_Q.shape)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        # print("W_K shape ", W_K.shape)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        # print("W_V shape ", W_V.shape)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        # note that model_state_dict[f"_orig_mod.transformer.h.{l}.attn.bias"] is actually storing the causal mask
        # not the attention bias
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = gpt2_rope[f"transformer.h.{l}.attn.c_proj.weight"]
        # print("W_O shape", W_O.shape)
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = gpt2_rope[f"transformer.h.{l}.ln_2.weight"]
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        W_in = gpt2_rope[f"transformer.h.{l}.mlp.c_fc.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        W_out = gpt2_rope[f"transformer.h.{l}.mlp.c_proj.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)


    state_dict["ln_final.w"] = gpt2_rope["transformer.ln_f.weight"]
    state_dict["ln_final.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    W_U = gpt2_rope["lm_head.weight"]
    state_dict["unembed.W_U"] = W_U.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict
