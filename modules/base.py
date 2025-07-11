import torch
import torch.nn as nn
from collections import OrderedDict

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)  # Base LN without affine params
        self.fc_gamma = nn.Linear(cond_dim, num_features)
        self.fc_beta = nn.Linear(cond_dim, num_features)

    def forward(self, x, cond):
        x = self.norm(x)
        gamma = self.fc_gamma(cond).unsqueeze(0)
        beta = self.fc_beta(cond).unsqueeze(0)
        return gamma * x + beta
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio = 4.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm, adaln=False):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 6 * d_model, bias=True)
            )
            self.ln_1 = norm_layer(d_model, elementwise_affine=False)
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            if adaln: self.ln_2 = norm_layer(d_model, elementwise_affine=False)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(self, x: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor, attn_mask=None, adaln_timestep_cond=None):
        if attn_mask is not None: attn_mask = attn_mask.repeat(self.n_head, 1, 1)
        if adaln_timestep_cond is None:
            if attn_mask is not None:
                attn_output = self.attention(self.ln_1(x), attn_mask=attn_mask)
            else: attn_output = self.attention(self.ln_1(x))
            
            x = x + attn_output
            if self.mlp_ratio > 0: 
                x = x + self.mlp(self.ln_2(x))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_timestep_cond).chunk(6, dim=1)
            if attn_mask is not None:
                attn_output = gate_msa.unsqueeze(0) * self.attention(modulate(self.ln_1(x), shift_msa, scale_msa), attn_mask=attn_mask)
            else: attn_output = gate_msa.unsqueeze(0) * self.attention(modulate(self.ln_1(x), shift_msa, scale_msa))
            
            x = x + attn_output
            if self.mlp_ratio > 0: 
                x = x + gate_mlp.unsqueeze(0) * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
            
        return x



