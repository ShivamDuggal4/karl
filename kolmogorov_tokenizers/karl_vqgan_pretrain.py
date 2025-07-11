import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, random
from modules.base import _expand_token
from modules.latent_distillation_modules import Encoder, Decoder

class AdaptiveLengthImageTokenizer(nn.Module):
    def __init__(self, 
            base_tokenizer,
            encoder_width, encoder_num_layers, encoder_num_heads,
            decoder_width, decoder_num_layers, decoder_num_heads,
            quantize_latent=True, factorize_latent=True, vq_codebook_size=4096, vq_token_dim=12, vq_commitment_cost=0.25, vq_use_l2_norm = True,
            num_init_latent_tokens=32, img_size=256, patch_size=16, max_rollout_iters=8,
            dynamic_halting=True, dynamic_halting_threshold=0.55, max_grid_tokens=64,
            train_stage="latent_distillation_pretrain"
        ):
        
        super().__init__()
        
        self.min_tokens = 16
        self.max_tokens = 512
        self.train_stage = train_stage
        self.quantize_latent = quantize_latent
        if quantize_latent is True: factorize_latent=True
        self.factorize_latent = factorize_latent
        self.dynamic_halting = dynamic_halting
        self.dynamic_halting_threshold = dynamic_halting_threshold
        # self.max_rollout_iters = max_rollout_iters
        # grid_size = img_size // patch_size
        scale = encoder_width ** -0.5

        self.encoder_ln_pre = nn.LayerNorm(encoder_width)
        self.encoder_ln_post_halt = nn.LayerNorm(encoder_width)
        self.encoder_ln_post = nn.LayerNorm(encoder_width)
        self.pre_quantizer_mlp = nn.Linear(encoder_width, vq_token_dim, bias=True)
        self.halting_mlp = nn.Sequential(
            nn.Linear(encoder_width, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, 1, bias=True)
        )

        self.encoder = Encoder(encoder_width, encoder_num_layers, encoder_num_heads, adaln=False)
        self.encoder_positional_embedding = nn.Parameter(scale * torch.randn(encoder_width, max_grid_tokens, max_grid_tokens))
        
        self.decoder = Decoder(decoder_width, decoder_num_layers, decoder_num_heads, factorize_latent=self.factorize_latent, factorized_latent_dim=vq_token_dim, output_dim=base_tokenizer.codebook_size, adaln=False)
        self.decoder_positional_embedding = nn.Parameter(scale * torch.randn(decoder_width, max_grid_tokens, max_grid_tokens))
        self.decoder_mask_token  = nn.Parameter(scale * torch.randn(1, 1, decoder_width))
        self.decoder_embed = nn.Linear(12, decoder_width, bias=True)
        self.decoder_latent_tokens_timestep_embed = nn.Parameter(scale * torch.randn(512, decoder_width))
        
        self.latent_tokens = nn.Parameter(scale * torch.randn(512, encoder_width))
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=encoder_width-base_tokenizer.embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True)

        # self.all_token_counts = torch.arange(0, 256+32, 32)[1:]
        # self.all_token_counts = torch.cat((torch.arange(1, 16, 1), torch.arange(0, 256+128, 16)[1:]), dim=0)
        self.all_token_counts = torch.arange(0, 256+128, 16)[1:] ## original
        print("all_token_counts: ", self.all_token_counts)
        
        self.small_code_losses = torch.tensor([0.05])

        self.code_losses = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.])
        self.nll_losses = torch.tensor([0.1, 0.5, 1., 2., 3., 4., 5., 6., 7.])
        self.rec_losses = torch.tensor([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.38])
        self.code_loss_embeddings = nn.Parameter(self.code_losses[:, None].repeat(1, encoder_width), requires_grad=True)
        self.nll_loss_embeddings = nn.Parameter(self.nll_losses[:, None].repeat(1, encoder_width), requires_grad=True)
        self.rec_loss_embeddings = nn.Parameter(self.rec_losses[:, None].repeat(1, encoder_width), requires_grad=True)

        initial_mask = self.rec_losses <= 0.04
        self.logged_min_loss_values_prob = torch.zeros_like(self.rec_losses)
        self.logged_min_loss_values_prob[initial_mask] = 1.0
        self.logged_min_loss_values_prob /= self.logged_min_loss_values_prob.sum()

        self.apply(self._init_weights)
        
        if self.quantize_latent:
            from modules.vector_quantizer import VectorQuantizer
            # Intialization for Quantizer is done inside VectorQuantizer
            self.quantize = VectorQuantizer(
                codebook_size=vq_codebook_size,
                token_size=vq_token_dim,
                commitment_cost=vq_commitment_cost,
                use_l2_norm=vq_use_l2_norm)
        
        self.base_tokenizer = base_tokenizer

        # TODO: Different loss weights per iteration might not be very critical
        self.lambda_loss_weight = [2.5, 2.0, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0]
        
        if self.train_stage=="full_finetuning":
            # TODO: Ablate the requirement of different discriminators for different recurrent rollout iterations.
            # Intuition is at different rollout iteration .....
            from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
            self.gan_losses = nn.ModuleList([VQLPIPSWithDiscriminator(
                disc_conditional= False, disc_in_channels= 3, 
                disc_start= 0, disc_weight= 0.2, codebook_weight= 1.0, # perceptual_weight=0.0
            ) for _ in range(self.max_rollout_iters)])
        
        if self.train_stage=="latent_distillation_pretrain":
            from modules.losses.nll import LabelSmoothingCrossEntropy
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def preprocess_encoder(self, img_tokens, grid_shape_2d):
        # x = img_tokens
        sampled_positional_embeddings = F.interpolate(self.encoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.encoder_positional_embedding.shape[0], -1).permute(0,2,1)
        # x = x + sampled_positional_embeddings
        # return x, sampled_positional_embeddings
        return sampled_positional_embeddings
    
    def preprocess_decoder(self, img_tokens, grid_shape_2d):
        mask_tokens = self.decoder_mask_token.repeat(img_tokens.shape[0], img_tokens.shape[1], 1).to(img_tokens.dtype)
        sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
        mask_tokens = mask_tokens + sampled_positional_embeddings
        return mask_tokens


    def reconstruct_images(self, logits, code, reconstruction_shape):
        code = code.reshape(code.shape[0], reconstruction_shape[0], reconstruction_shape[1], code.shape[-1]).permute([0,3,1,2])
        return self.base_tokenizer.vqgan.decode(code)
        if self.train_stage=="latent_distillation_pretrain":
            # decode using logits.
            logits = logits[:, :, :self.base_tokenizer.codebook_size]
            sample_dist = torch.distributions.categorical.Categorical(logits=logits)
            sampled_ids = sample_dist.sample()
            bsz = sampled_ids.shape[0]
            z_q = self.base_tokenizer.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, reconstruction_shape[0], reconstruction_shape[1], self.base_tokenizer.codebook_emb_dim))
            return self.base_tokenizer.vqgan.decode(z_q)
        elif self.train_stage=="full_finetuning":
            # decode using code directly.
            code = code.reshape(code.shape[0], reconstruction_shape[0], reconstruction_shape[1], code.shape[-1]).permute([0,3,1,2])
            return self.base_tokenizer.vqgan.decode(code)

    def get_2d_tokens(self, imgs):
        vqgan_tokens, gt_indices = self.base_tokenizer.get_img_tokens(imgs)
        img_tokens = self.patch_embed(imgs)
        grid_shape_2d = img_tokens.shape[-2:]
        img_tokens = img_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1])
        vqgan_tokens = vqgan_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1])
        img_tokens = torch.cat((img_tokens, vqgan_tokens), dim=2)
        # img_tokens = F.normalize(img_tokens, dim=-1)
        return img_tokens, gt_indices, grid_shape_2d

    def get_positional_embeddings(self, img_tokens, grid_shape_2d):
        assert(grid_shape_2d==(16,16) or grid_shape_2d==(16,32))
        masked_2d_tokens = self.preprocess_decoder(img_tokens, grid_shape_2d)
        encoder_sampled_positional_embeddings = self.preprocess_encoder(img_tokens, grid_shape_2d)
        # num_img_tokens = img_tokens.shape[1]
        # x = torch.cat([img_tokens, init_latent_tokens + self.timestep_embedding[0]], dim=1)
        # x = self.encoder_ln_pre(x)
        return masked_2d_tokens, encoder_sampled_positional_embeddings


    def sample_token_counts(self, tokens_to_sample, batch_size, sampled_token_counts=None, sampled_code_losses=None, min_token_to_add=0, predefined_token_counts=None, predefined_reconstruction_loss=None):
        if predefined_token_counts is None:
            perm_indices = torch.randperm(self.all_token_counts.shape[0])
            token_counts = self.all_token_counts[perm_indices[:tokens_to_sample]].tolist()
            if sampled_token_counts is not None: 
                # token_counts = [sampled_token_counts[idx]+max(min_token_to_add, token_counts[idx]) for idx in range(len(sampled_token_counts))]
                token_counts = [256 for idx in range(len(sampled_token_counts))]
            else:
                # token_counts = [32 if count==0 else count for count in token_counts]
                token_counts = [16 if count==0 else count for count in token_counts] ##original
                # token_counts = [count + torch.randint(0, 16, (1,)).item() for count in token_counts]
            token_counts = [min(count, 512) for count in token_counts]
            # token_counts = [min(count, 256) for count in token_counts]
        else:
            token_counts = predefined_token_counts
        print("token_counts: ", token_counts)
        
        if sampled_code_losses is None:
            code_loss_idx = torch.randint(0, self.code_losses[0:1].shape[0], size=(batch_size, len(token_counts))).reshape(-1)
            code_losses = self.small_code_losses[code_loss_idx].reshape(batch_size, len(token_counts))
            code_losses = code_losses[:,:,None]
            # rec_losses = torch.zeros_like(code_losses) + predefined_reconstruction_loss
            if predefined_reconstruction_loss is None: 
                indices = torch.multinomial(self.logged_min_loss_values_prob, batch_size*len(token_counts), replacement=True)
                predefined_reconstruction_loss = self.rec_losses[indices].reshape(batch_size, len(token_counts), 1)
                rec_losses = predefined_reconstruction_loss
                print(rec_losses[:10], "rec_losses iteration 0")
                print(self.logged_min_loss_values_prob, "logged_min_loss_values_prob")
            else:
                rec_losses = torch.zeros_like(code_losses) + predefined_reconstruction_loss
            
            nll_losses = torch.zeros_like(code_losses) + self.nll_losses[0:1]
            code_losses = torch.cat((code_losses, rec_losses, nll_losses), dim=-1)
        else:
            code_losses = sampled_code_losses
        # print("code_losses all iters: ", code_losses[0], code_losses.shape)
        return token_counts, code_losses
    

    def get_discretized_loss_and_embedding(self, code_loss_tensor, rec_loss_tensor, nll_loss_tensor):
        """
        Discretizes code and nll loss tensors into the closest bins from
        self.code_losses and self.nll_losses, and returns corresponding embeddings.

        Args:
            code_loss_tensor (torch.Tensor): shape [B, S]
            nll_loss_tensor (torch.Tensor): shape [B, S]

        Returns:
            binned_code_loss (torch.Tensor): shape [B, S]
            binned_code_embed (torch.Tensor): shape [B, S, D]
            binned_nll_loss (torch.Tensor): shape [B, S]
            binned_nll_embed (torch.Tensor): shape [B, S, D]
        """
        def discretize(loss_tensor, bin_values, bin_embeddings):
            B, S = loss_tensor.shape
            bin_values = bin_values.to(loss_tensor.device)
            bin_embeddings = bin_embeddings.to(loss_tensor.device)

            loss_tensor_exp = loss_tensor.unsqueeze(-1)         # [B, S, 1]
            bin_values_exp = bin_values.view(1, 1, -1)           # [1, 1, N]

            valid_bins_mask = bin_values_exp >= loss_tensor_exp
            bin_diffs = bin_values_exp - loss_tensor_exp
            bin_diffs[~valid_bins_mask] = float('inf')

            bin_indices = bin_diffs.argmin(dim=-1)              # [B, S]

            # Fix: fallback to last bin if no valid bin exists
            all_invalid = ~valid_bins_mask.any(dim=-1)          # [B, S]
            bin_indices[all_invalid] = bin_values.shape[0] - 1

            binned_loss = bin_values[bin_indices]               # [B, S]
            binned_embed = bin_embeddings[bin_indices]          # [B, S, D]

            return binned_loss, binned_embed

        binned_code_loss, binned_code_embed = discretize(
            code_loss_tensor, self.code_losses, self.code_loss_embeddings
        )
        binned_nll_loss, binned_nll_embed = discretize(
            nll_loss_tensor, self.nll_losses, self.nll_loss_embeddings
        )
        binned_rec_loss, binned_rec_embed = discretize(
            rec_loss_tensor, self.rec_losses, self.rec_loss_embeddings
        )

        return torch.cat((binned_code_loss[...,None], binned_rec_loss[...,None], binned_nll_loss[...,None]), dim=-1), torch.cat((binned_code_embed[:,:,None], binned_rec_embed[:,:,None], binned_nll_embed[:,:,None]), dim=2)
        # return binned_code_loss, binned_code_embed, binned_nll_loss, binned_nll_embed


    def encode(self, imgs, predefined_token_counts=None, predefined_reconstruction_loss=None,
            return_min_length_embedding=True, 
            token_selection_criteria="reconstruction_loss", threshold=0.07, 
            return_embedding_type="latent_tokens"):
        
        # parameter return_all_embeddings returns multiple representations per image.
        # parameter return_min_length_embedding returns smallest length embedding with satisfies an objective (reconstruction loss < threshold for now).
        # parameter return_embedding_type \in ["latent_tokens", "image_and_latent_all_tokens", "image_tokens"], default="latent_tokens"
        
        # token selection criteria decides the satisfyable length of the embedding.
        # right now we only support reconstruction_loss as the automatic token selection criteria.
        # alternative TSC used in the paper require oracle / GT depth or class labels.
        # one could also learn a token selection criteria based on input image (we might release this at some point)

        reconstruction_iters = []
        if return_min_length_embedding: 
            assert(return_embedding_type=="latent_tokens")
            best_tsc, best_tsc_iter = torch.inf, -1 # tsc = token selection criteria                 
        
        _, all_logs = self.forward(imgs, predefined_token_counts=predefined_token_counts, predefined_reconstruction_loss=predefined_reconstruction_loss)
        all_embeddings = []
        all_reconstructions = []
        for iter, iter_logs_dict in enumerate(all_logs):
            for key in iter_logs_dict.keys():
                if "reconstructed_imgs" in key:
                    reconstructed_imgs = iter_logs_dict[key]
                    loglaplace_loss = torch.abs(reconstructed_imgs - imgs).mean()
                    print("key: {} | token count: {} | loglaplace_loss: {}".format(key, key.split("_")[-1], loglaplace_loss))  
                    all_reconstructions.append(iter_logs_dict[key])
    
    
        return all_embeddings, all_reconstructions, all_logs

    def forward_single_image(self, imgs, predefined_token_counts=None, predefined_reconstruction_loss=None):
        print("forward single_image ")
        device = imgs.get_device()
        all_logs = []
        sampled_token_counts, sampled_code_losses = self.sample_token_counts(tokens_to_sample=1, batch_size=imgs.shape[0], predefined_token_counts=predefined_token_counts, predefined_reconstruction_loss=predefined_reconstruction_loss)
        sampled_code_losses, sampled_code_loss_embeddings = self.get_discretized_loss_and_embedding(sampled_code_losses[...,0], sampled_code_losses[...,1], sampled_code_losses[...,2])

        img_tokens, gt_indices, grid_shape_2d = self.get_2d_tokens(imgs)
        total_loss, logs, code_losses, encoded_latent_tokens, encoder_attn_mask = self.forward_call(imgs,
            img_tokens, gt_indices, grid_shape_2d, is_latent_halting=False,
            sampled_token_counts=sampled_token_counts, sampled_code_losses=None, sampled_code_loss_embeddings=sampled_code_loss_embeddings,
        )
        all_logs += logs

        # sampled_token_counts_2 = sampled_token_counts
        sampled_token_counts_2, sampled_code_losses = self.sample_token_counts(tokens_to_sample=1, batch_size=imgs.shape[0], sampled_token_counts=sampled_token_counts, sampled_code_losses=torch.stack(code_losses, dim=1))
        # sampled_token_count_delta = [max(0, sampled_token_counts_2[idx]-max(32,sampled_token_counts[idx])) for idx in range(len(sampled_token_counts_2))]
        sampled_token_count_delta = [max(0, sampled_token_counts_2[idx]-sampled_token_counts[idx]) for idx in range(len(sampled_token_counts_2))]
        sampled_code_losses, sampled_code_loss_embeddings = self.get_discretized_loss_and_embedding(sampled_code_losses[...,0], sampled_code_losses[...,1], sampled_code_losses[...,2])
        total_loss, logs, code_losses, encoded_latent_tokens, encoder_attn_mask = self.forward_call(imgs,
            img_tokens, gt_indices, grid_shape_2d, is_latent_halting=True,
            sampled_token_counts=sampled_token_counts_2, sampled_code_losses=sampled_code_losses.to(device), #sampled_code_losses=sampled_code_losses[...,0].to(device), 
            sampled_code_loss_embeddings=sampled_code_loss_embeddings,
            sampled_token_count_delta=sampled_token_count_delta,
            total_loss=total_loss
        )
        all_logs += logs
        return total_loss, all_logs


    def forward(self, imgs, epoch=None, gan_optimizer_idx=None, gan_loss_weight=None, predefined_token_counts=None, predefined_reconstruction_loss=None):
        self.epoch = epoch
        self.gan_optimizer_idx = gan_optimizer_idx
        self.gan_loss_weight = gan_loss_weight
        total_loss, all_logs = self.forward_single_image(imgs, predefined_token_counts=predefined_token_counts, predefined_reconstruction_loss=predefined_reconstruction_loss)
        print("=================================")
        return total_loss, all_logs


    def forward_call(self, imgs, img_tokens, gt_indices, grid_shape_2d, is_latent_halting,
            sampled_token_counts, sampled_code_losses, sampled_code_loss_embeddings, sampled_token_count_delta=None, total_loss=0., aug_id=None,
            encoded_latent_tokens=None, encoder_attn_mask=None):
        
        all_logs = []
        masked_2d_tokens, encoder_sampled_positional_embeddings = self.get_positional_embeddings(img_tokens, grid_shape_2d)
        total_loss, logs, code_losses, encoded_latent_tokens, encoder_attn_mask = self.forward_executer(imgs,
            img_tokens, encoder_sampled_positional_embeddings, masked_2d_tokens, gt_indices, 
            is_latent_halting=is_latent_halting, total_loss=total_loss,
            sampled_token_counts=sampled_token_counts, sampled_code_losses=sampled_code_losses, sampled_code_loss_embeddings=sampled_code_loss_embeddings, 
            sampled_token_count_delta=sampled_token_count_delta, grid_shape_2d=grid_shape_2d, aug_id=aug_id,
            input_latent_tokens=encoded_latent_tokens, input_attn_mask=encoder_attn_mask
        )
        all_logs += logs
        return total_loss, all_logs, code_losses, encoded_latent_tokens, encoder_attn_mask
        
    def prepare_decoder_latents(self, latent_embeddings, num_img_tokens, is_latent_halting, halting_prob=0.75, detach=True):
        # latent_embeddings = self.encoder_ln_post(latent_embeddings)
        # latent_tokens_halt_logits = self.halting_mlp(latent_embeddings)
        # latent_tokens_halt_logits = self.halting_mlp(self.encoder_ln_post_halt(-1. * latent_embeddings))
        if detach: 
            assert(False)
            # print("epoch: {} | detach: {} detaching...".format(self.epoch, detach))
            latent_tokens_halt_logits = self.halting_mlp(self.encoder_ln_post_halt(latent_embeddings.detach()))
        else: 
            # print("epoch: {} | detach: {} not detaching...".format(self.epoch, detach))
            latent_tokens_halt_logits = self.halting_mlp(self.encoder_ln_post_halt(latent_embeddings))
        # latent_tokens_halt_logits = self.halting_mlp(self.encoder_ln_post_halt(latent_embeddings))
        latent_tokens_halt_prob = F.sigmoid(latent_tokens_halt_logits)
        latent_tokens_factorized = self.pre_quantizer_mlp(self.encoder_ln_post(latent_embeddings))
        # latent_tokens_factorized = self.pre_quantizer_mlp(latent_embeddings)    
        quantize_token_mask = None
        attn_mask = None
        
        if not is_latent_halting:
            return latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, quantize_token_mask, attn_mask

        bs = latent_embeddings.shape[0]
        device = latent_embeddings.get_device()
        num_latent_tokens = latent_tokens_factorized.shape[1]
        num_all_tokens = num_img_tokens+num_latent_tokens
        
        img_token_mask = torch.zeros(bs, num_img_tokens, dtype=torch.bool, device=device)
        latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>halting_prob)
        combined_mask = torch.cat([img_token_mask, latent_tokens_halt_prob_mask.bool()[...,0]], dim=1)  # shape: (bs, num_all_tokens)
        attn_mask = combined_mask.unsqueeze(2) | combined_mask.unsqueeze(1)  # shape: (bs, num_all_tokens, num_all_tokens)
        diag_idx = torch.arange(num_all_tokens, device=device)
        attn_mask[:, diag_idx, diag_idx] = False
        # print(attn_mask[0,256:276,256:276], "attn_mask")

        # latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>halting_prob)*1.
        # latent_tokens_halt_prob_mask = latent_tokens_halt_prob + (latent_tokens_halt_prob_mask-latent_tokens_halt_prob).detach()
        # latent_tokens_factorized_masked = latent_tokens_factorized * (1-latent_tokens_halt_prob_mask) + self.latent_mask_token[None, None] * latent_tokens_halt_prob_mask
        # latent_tokens_factorized = latent_tokens_factorized + (latent_tokens_factorized_masked - latent_tokens_factorized).detach()
        return latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, quantize_token_mask, attn_mask
        

    def forward_executer(self, imgs,
            img_tokens, encoder_sampled_positional_embeddings, masked_2d_tokens, gt_indices, is_latent_halting, total_loss,
            sampled_token_counts, sampled_code_losses, sampled_code_loss_embeddings, sampled_token_count_delta, grid_shape_2d, aug_id,
            input_latent_tokens=None, input_attn_mask=None):

        all_iter_code_loss = []
        all_logs = []
        encoded_latent_tokens = []
        encoder_attn_mask = []
        for iter in range(len(sampled_token_counts)):
            curr_token_count = sampled_token_counts[iter]
            pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:curr_token_count]
            start_token_id = 0

            x = img_tokens + encoder_sampled_positional_embeddings
            if input_latent_tokens is None:
                x = torch.cat([x, (self.latent_tokens[start_token_id:curr_token_count])[None].repeat(x.shape[0], 1, 1)], dim=1)
            else:
                assert(is_latent_halting is False)
                x = torch.cat([x, input_latent_tokens[iter]], dim=1)
            # x = torch.cat([x, (self.latent_tokens[start_token_id:curr_token_count] + self.timestep_embedding[iter,start_token_id:curr_token_count])[None].repeat(x.shape[0], 1, 1)], dim=1)
            x = torch.cat((sampled_code_loss_embeddings[:,iter,1:2].to(x.get_device()), x), dim=1)
            # x = torch.cat((self.masking_embedding[(is_latent_halting)*1][None,None].repeat(x.shape[0],1,1), sampled_code_loss_embeddings[:,iter,0:1].to(device), x), dim=1)
            # x = torch.cat((self.masking_embedding[(is_latent_halting)*1][None,None].repeat(x.shape[0],1,1), sampled_code_loss_embeddings[:,iter,0:2].to(device), x), dim=1)
            # x = torch.cat((self.masking_embedding[(is_latent_halting)*1][None,None].repeat(x.shape[0],1,1), sampled_code_loss_embeddings[:,iter,1:2].to(x.get_device()), x), dim=1)
            x = self.encoder_ln_pre(x)
            if input_attn_mask is None: attn_mask=None
            else: attn_mask = input_attn_mask[iter]
            x = self.encoder(x, attn_mask=attn_mask, adaln_timestep_cond=None)
            # x = self.encoder(x, attn_mask=None, adaln_timestep_cond=sampled_code_loss_embeddings[:,iter])
            # if is_latent_halting:
            #     x = self.encoder(x, attn_mask=None, adaln_timestep_cond=None)
            # else:
            #     x = self.encoder_unmasked(x, attn_mask=None, adaln_timestep_cond=None)
            encoded_latent_tokens.append(x[:, 1+img_tokens.shape[1]:])
            latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, quantize_token_count, attn_mask = \
                self.prepare_decoder_latents(x[:, 1+img_tokens.shape[1]:], num_img_tokens=1+img_tokens.shape[1], is_latent_halting=is_latent_halting, detach=False) #(self.epoch<=40))
                # self.prepare_decoder_latents(x[:, 2+img_tokens.shape[1]:], num_img_tokens=img_tokens.shape[1], is_latent_halting=is_latent_halting, detach=False) #(self.epoch<=40))
                # self.prepare_decoder_latents(x[:, 3+img_tokens.shape[1]:], num_img_tokens=img_tokens.shape[1], is_latent_halting=is_latent_halting, detach=False) #detach=(self.epoch<=40))
                # self.prepare_decoder_latents(x[:, img_tokens.shape[1]:], num_img_tokens=img_tokens.shape[1], is_latent_halting=is_latent_halting)
                
            if input_attn_mask is not None: attn_mask = input_attn_mask[iter]
            if attn_mask is not None: 
                decoder_attn_mask = attn_mask[:,1:,1:]
                encoder_attn_mask.append(attn_mask)
            else: decoder_attn_mask = None
            iter_logs_dict = {}
            if self.quantize_latent:
                print("quantization")
                latent_tokens_quantized, quant_result_dict = self.quantize(latent_tokens_factorized, is_quantize=True, token_count=quantize_token_count)
            else:
                print("no quantization")
                latent_tokens_quantized = latent_tokens_factorized
            decoded_latent_1D_tokens = self.decoder_embed(latent_tokens_quantized)
            decoded_logits = self.decoder(decoded_latent_1D_tokens, masked_2d_tokens, pos_embed_indices, attn_mask=decoder_attn_mask) 
            decoded_logits_softmax = torch.nn.functional.softmax(decoded_logits, dim=-1)
            decoded_code = torch.einsum('nlc,cd->nld', decoded_logits_softmax, self.base_tokenizer.vqgan.quantize.embedding.weight.data)            
            if self.train_stage == "latent_distillation_pretrain" and not is_latent_halting:
                with torch.no_grad():
                    reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                    iter_rec_loss = torch.abs(imgs.contiguous() - reconstructed_imgs.contiguous()).reshape(decoded_code.shape[0], -1).mean(dim=-1)
                    print("iter_rec_loss: {} | iter_rec_loss_shape: {}".format(iter_rec_loss, iter_rec_loss.shape))

                    decay = 0.99 #0.8
                    batch_min = iter_rec_loss.min().item()
                    harder_or_equal_mask = self.rec_losses >= batch_min
                    self.logged_min_loss_values_prob[harder_or_equal_mask] *= decay
                    self.logged_min_loss_values_prob /= self.logged_min_loss_values_prob.sum()

            latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>0.75)*1.
            if is_latent_halting:
                masked_token_count = sampled_token_count_delta[iter]
                print(masked_token_count, "masked_token_count")
                # latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>0.75)*1.
                # latent_token_halt_loss = F.relu(0.75 - latent_tokens_halt_prob[:,-masked_token_count:]).mean()
                # latent_token_active_loss = F.relu(latent_tokens_halt_prob[:,:-masked_token_count] - 0.75).mean()

                # with torch.cuda.amp.autocast(enabled=False):
                # latent_token_halt_loss = ((1.*latent_tokens_factorized.shape[1])/masked_token_count) * F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,-masked_token_count:], torch.ones_like(latent_tokens_halt_prob[:,-masked_token_count:]))
                # latent_token_active_loss = ((1.*latent_tokens_factorized.shape[1])/(latent_tokens_factorized.shape[1]-masked_token_count)) * F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,:-masked_token_count], torch.zeros_like(latent_tokens_halt_prob[:,:-masked_token_count]))
                if masked_token_count==0:
                    latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits, torch.zeros_like(latent_tokens_halt_prob))
                    latent_token_halt_loss = None
                else:
                    latent_token_halt_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,-masked_token_count:], torch.ones_like(latent_tokens_halt_prob[:,-masked_token_count:]))
                    latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,:-masked_token_count], torch.zeros_like(latent_tokens_halt_prob[:,:-masked_token_count]))
                    total_loss = total_loss + latent_token_halt_loss
                    if aug_id is not None:
                        iter_logs_dict.update({
                            "latent_token_halt_loss_bce_{}_{}".format(latent_tokens_factorized.shape[1], aug_id): latent_token_halt_loss.item(),
                        })
                    else:
                        iter_logs_dict.update({
                            "latent_token_halt_loss_bce_{}".format(latent_tokens_factorized.shape[1]): latent_token_halt_loss.item(),
                        })
                
                # all_halted_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits, torch.ones_like(latent_tokens_halt_prob))
                total_loss = total_loss + latent_token_active_loss #+ 0.1*all_halted_loss
                print("total: {}| masked: {} | halt_loss: {} | active_loss: {} | mask_count: {} | active_count: {}".format(latent_tokens_factorized.shape[1], masked_token_count, latent_token_halt_loss, latent_token_active_loss, latent_tokens_halt_prob_mask[...,0].sum(dim=-1).mean(), (latent_tokens_factorized.shape[1] - latent_tokens_halt_prob_mask[...,0].sum(dim=-1)).mean()))
                # print(latent_tokens_halt_prob.shape, latent_tokens_halt_prob[...,0], "latent_tokens_halt_prob")
                if aug_id is not None:
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}_{}".format(latent_tokens_factorized.shape[1], aug_id): latent_token_active_loss.item(),
                        # "latent_token_all_halted_loss_bce_{}_{}".format(latent_tokens_factorized.shape[1], aug_id): all_halted_loss.item(),
                    })
                else:
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}".format(latent_tokens_factorized.shape[1]): latent_token_active_loss.item(),
                        # "latent_token_all_halted_loss_bce_{}".format(latent_tokens_factorized.shape[1]): all_halted_loss.item(),
                    })
            else:
                masked_token_count = 256 - latent_tokens_factorized.shape[1]
                latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits, torch.zeros_like(latent_tokens_halt_prob))
                total_loss = total_loss + 1. * latent_token_active_loss
                print("total: {}| active_loss: {} | mask_count: {} | active_count: {}".format(latent_tokens_factorized.shape[1], latent_token_active_loss, latent_tokens_halt_prob_mask[...,0].sum(dim=-1).mean(), (latent_tokens_factorized.shape[1] - latent_tokens_halt_prob_mask[...,0].sum(dim=-1)).mean()))
                if aug_id is not None:
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}_0_{}".format(latent_tokens_factorized.shape[1], aug_id): latent_token_active_loss.item(),
                    })
                else:
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}_0".format(latent_tokens_factorized.shape[1]): latent_token_active_loss.item(),
                    })

            if self.train_stage == "latent_distillation_pretrain":
                if is_latent_halting is False: loss_capping = None
                else: loss_capping = sampled_code_losses[:,iter]
                iter_nll_loss, iter_code_loss, iter_code_loss_batch, iter_nll_loss_batch = self.forward_loss(gt_indices, decoded_logits, decoded_code, loss_capping)
                print("nll_loss: {} | code_loss: {}".format(iter_nll_loss, iter_code_loss))
                if is_latent_halting is False: 
                    all_iter_code_loss.append(torch.cat((iter_code_loss_batch.detach()[...,None], iter_rec_loss[...,None], iter_nll_loss_batch.detach()[...,None]), dim=-1))
                    # all_iter_code_loss.append(iter_code_loss_batch.detach())
                    # all_iter_nll_loss.append(iter_nll_loss_batch.detach())
                total_loss = total_loss + (iter_nll_loss + 1. * iter_code_loss)
                if aug_id is not None:
                    iter_logs_dict.update({
                        "nll_loss_{}_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1, aug_id): iter_nll_loss.item(),
                        "code_loss_{}_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1, aug_id): iter_code_loss.item(),
                    })
                else:
                    iter_logs_dict.update({
                        "nll_loss_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): iter_nll_loss.item(),
                        "code_loss_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): iter_code_loss.item(),
                    })

            elif self.train_stage == "full_finetuning":
                if is_latent_halting is False: loss_capping = None
                else: loss_capping = sampled_code_losses[:,iter]
                reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                gan_loss, logs_dict, iter_rec_loss = self.forward_gan_losses(imgs, reconstructed_imgs, loss_capping, is_latent_halting, aug_id, optimizer_idx=self.gan_optimizer_idx, latent_token_count=max(0,256-masked_token_count), discriminator_loss_weight=self.gan_loss_weight)
                print(iter_rec_loss, "iter_rec_loss")
                total_loss = total_loss + gan_loss
                iter_logs_dict.update(logs_dict)
                if is_latent_halting is False: 
                    all_iter_code_loss.append(iter_rec_loss[...,None])

                    decay = 0.99 #0.8
                    batch_min = iter_rec_loss.min().item()
                    harder_or_equal_mask = self.rec_losses >= batch_min
                    self.logged_min_loss_values_prob[harder_or_equal_mask] *= decay
                    self.logged_min_loss_values_prob /= self.logged_min_loss_values_prob.sum()

                if aug_id is not None:
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1, aug_id): reconstructed_imgs,
                    })
                else:
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): reconstructed_imgs,
                    })
                
            if self.quantize_latent: 
                total_loss = total_loss + (1. * quant_result_dict['quantizer_loss'])
                iter_logs_dict.update({
                    "quantization_loss_{}".format(latent_tokens_factorized.shape[1]): quant_result_dict['quantizer_loss'].item(),
                })
            
            
            if aug_id is not None:
                if  not self.training and "reconstructed_imgs_{}_{}".format(latent_tokens_factorized.shape[1], aug_id) not in iter_logs_dict:
                    reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}_{}".format(latent_tokens_factorized.shape[1], aug_id): reconstructed_imgs,
                    })  
            else:
                if  not self.training and "reconstructed_imgs_{}".format(latent_tokens_factorized.shape[1]) not in iter_logs_dict:
                    reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}".format(latent_tokens_factorized.shape[1]): reconstructed_imgs,
                    }) 
            
            all_logs.append(iter_logs_dict)

        # encoded_latent_tokens = torch.stack(encoded_latent_tokens, dim=0)
        # encoder_attn_mask = torch.stack(encoder_attn_mask, dim=0)
        if not self.training: return total_loss, all_logs, all_iter_code_loss, encoded_latent_tokens, encoder_attn_mask
        return total_loss, all_logs, all_iter_code_loss, encoded_latent_tokens, encoder_attn_mask
    

    def forward_loss(self, gt_indices, decoded_logits, decoded_code, loss_capping):
        bsz, seq_len = gt_indices.size()
        assert(bsz==decoded_code.shape[0])
        assert(seq_len==decoded_code.shape[1])
        
        nll_loss_pixel, _ = self.criterion(decoded_logits[:, :, :self.base_tokenizer.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        nll_loss_pixel = nll_loss_pixel.reshape(bsz, seq_len) #.mean(dim=-1)

        vqgan_embedding_shape = self.base_tokenizer.vqgan.quantize.embedding.weight.data.shape[-1]
        gt_code = torch.gather(self.base_tokenizer.vqgan.quantize.embedding.weight.data, dim=0, index=gt_indices.reshape(bsz*seq_len)[...,None].repeat(1, vqgan_embedding_shape))
        gt_code = gt_code.reshape(bsz, seq_len, vqgan_embedding_shape)
        assert(gt_code.shape == decoded_code.shape)
        code_loss = (gt_code - decoded_code)**2
        code_loss_pixel = code_loss.mean(dim=-1)
        code_loss_batch = code_loss.reshape(bsz, -1).mean(dim=-1)
        
        return nll_loss_pixel.mean(), code_loss.mean(), code_loss_batch, nll_loss_pixel.mean(dim=-1)



    def get_last_layer(self):
        return self.base_tokenizer.vqgan.decoder.conv_out.weight

    def forward_gan_losses(self, imgs, reconstructed_imgs, loss_capping, is_latent_halting, aug_id, optimizer_idx, latent_token_count, discriminator_loss_weight):
        assert(optimizer_idx is not None)
        # iter_idx = 0 #min(4, (latent_token_count // 64))
        iter_idx = max(0,min(8, (latent_token_count // 32))-1)
        if discriminator_loss_weight==0:
            global_step=-torch.inf
            self.gan_losses[iter_idx].discriminator_weight = 0.2
        else:
            global_step=torch.inf
            self.gan_losses[iter_idx].discriminator_weight = discriminator_loss_weight
        if optimizer_idx == 0:
            aeloss, log_dict_ae, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, loss_capping, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")

            iter_log_dict_ae = {}
            for key in log_dict_ae.keys():
                if aug_id is not None:
                    iter_log_dict_ae["{}_{}_{}_{}".format(key, latent_token_count, is_latent_halting*1, aug_id)] = log_dict_ae[key]
                else:
                    iter_log_dict_ae["{}_{}_{}".format(key, latent_token_count, is_latent_halting*1)] = log_dict_ae[key]
            
            return aeloss, iter_log_dict_ae, iter_rec_loss

        if optimizer_idx == 1:
            print("inside optimizer_idx==1 code")
            discloss, log_dict_disc, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, loss_capping, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")
            
            iter_log_dict_disc = {}
            for key in log_dict_disc.keys():
                iter_log_dict_disc["{}_{}".format(key, iter_idx)] = log_dict_disc[key]
            
            return discloss, iter_log_dict_disc, iter_rec_loss