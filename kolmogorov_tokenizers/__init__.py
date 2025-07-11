from omegaconf import OmegaConf
from modules.base_tokenizers import VQGANWrapper, LDMVAEWrapper

# KARL Latent-Distillation Encoder / Decoder configurations.
width      = {"small": 512, "base": 768, "semi_large": 1024}
num_layers = {"small": 8,   "base": 12,  "semi_large": 16}
num_heads  = {"small": 8,   "base": 12,  "semi_large": 16}

base_tokenizers = {
    "vqgan": VQGANWrapper,
    "vae": LDMVAEWrapper,
}

kolmogorov_tokenizers = {
    "vqgan": __import__('kolmogorov_tokenizers.karl_vqgan', fromlist=['KARLTokenizer']).KARLTokenizer,
    "vae": __import__('kolmogorov_tokenizers.karl_vae', fromlist=['KARLTokenizer']).KARLTokenizer,
}

kolmogorov_tokenizer_args = {
    "vqgan": OmegaConf.load('kolmogorov_tokenizers/configs/karl_vqgan.yaml'),
    "vae": OmegaConf.load('kolmogorov_tokenizers/configs/karl_vae.yaml')
}

def karl_tiny(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["small"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["small"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_tiny_semilarge_decoder(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["small"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_semilarge(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["semi_large"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_semilarge_opposite_heads(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_semilarge_small_heads(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_semilarge_semilarge_heads(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["semi_large"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_base(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["base"], decoder_num_layers=num_layers["base"], decoder_num_heads=num_heads["base"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_base_semilarge_feat(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["base"], decoder_num_heads=num_heads["semi_large"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_small_decoder_small(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["small"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_base(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["base"], encoder_num_layers=num_layers["base"], encoder_num_heads=num_heads["base"],
        decoder_width=width["base"], decoder_num_layers=num_layers["base"], decoder_num_heads=num_heads["base"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_base_decoder_small_semilarge_feat(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["base"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

# def karl_small_base_decoder(base_tokenizer_args, **kwargs):
#     base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
#     tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
#         base_tokenizer=base_tokenizer,
#         encoder_width=width["semilarge"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
#         decoder_width=width["semilarge"], decoder_num_layers=num_layers["base"], decoder_num_heads=num_heads["base"],
#         **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
#     )
#     return tokenizer

def karl_semilarge(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["semi_large"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["semi_large"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def karl_semilarge_decoder_small(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = kolmogorov_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["semi_large"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **kolmogorov_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer