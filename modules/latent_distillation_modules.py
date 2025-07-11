import torch
import torch.nn as nn
from modules.base import ResidualAttentionBlock

class Encoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, mlp_ratio=4.0, adaln=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio, adaln=adaln)
            )

    def forward(self, x, attn_mask=None, adaln_timestep_cond=None):
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask, adaln_timestep_cond)
        x = x.permute(1, 0, 2)
        return x  
    

class Decoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, factorize_latent, output_dim, mlp_ratio=4.0, adaln=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.factorize_latent = factorize_latent

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio, adaln=adaln))
        self.ln_post = nn.LayerNorm(width)

        self.ffn = nn.Sequential(
            nn.Linear(width, 2*width, bias=True), nn.Tanh(),
            nn.Linear(2*width, output_dim)
        )


    def forward(self, latent_1D_tokens, masked_2D_tokens, pos_embed_indices, adaln_timestep_cond=None, attn_mask=None):
        latent_1D_tokens = latent_1D_tokens + pos_embed_indices[None]
        x = torch.cat([masked_2D_tokens, latent_1D_tokens], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        all_attn_weights = []
        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask=attn_mask, adaln_timestep_cond=adaln_timestep_cond)
        x = x.permute(1, 0, 2)

        reconstructed_2D_tokens = x[:, :masked_2D_tokens.shape[1]]
        reconstructed_2D_tokens = self.ln_post(reconstructed_2D_tokens)
        reconstructed_2D_tokens = self.ffn(reconstructed_2D_tokens)

        return reconstructed_2D_tokens