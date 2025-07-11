import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.latent_distillation_modules import Encoder, Decoder

class KARLTokenizer(nn.Module):
    def __init__(self, 
            base_tokenizer,
            encoder_width, encoder_num_layers, encoder_num_heads,
            decoder_width, decoder_num_layers, decoder_num_heads,
            quantize_latent=True, factorize_latent=True, vq_codebook_size=4096, vq_token_dim=12, vq_commitment_cost=0.25, vq_use_l2_norm = True,
            num_init_latent_tokens=32, patch_size=16, max_rollout_iters=8,
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
        scale = encoder_width ** -0.5

        self.encoder = Encoder(encoder_width, encoder_num_layers, encoder_num_heads, adaln=False)
        self.encoder_positional_embedding = nn.Parameter(scale * torch.randn(encoder_width, max_grid_tokens, max_grid_tokens))
        self.encoder_ln_pre = nn.LayerNorm(encoder_width)
        self.encoder_ln_post_halt = nn.LayerNorm(encoder_width)
        self.encoder_ln_post = nn.LayerNorm(encoder_width)
        self.pre_quantizer_mlp = nn.Linear(encoder_width, vq_token_dim, bias=True)
        self.halting_mlp = nn.Sequential(nn.Linear(encoder_width, 512, bias=True), nn.Tanh(), nn.Linear(512, 1, bias=True))

        self.decoder = Decoder(decoder_width, decoder_num_layers, decoder_num_heads, factorize_latent=self.factorize_latent, output_dim=base_tokenizer.codebook_size, adaln=False)
        self.decoder_positional_embedding = nn.Parameter(scale * torch.randn(decoder_width, max_grid_tokens, max_grid_tokens))
        self.decoder_mask_token  = nn.Parameter(scale * torch.randn(1, 1, decoder_width))
        self.decoder_embed = nn.Linear(vq_token_dim, decoder_width, bias=True)
        self.decoder_latent_tokens_timestep_embed = nn.Parameter(scale * torch.randn(512, decoder_width))
        
        self.latent_tokens = nn.Parameter(scale * torch.randn(512, encoder_width))
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=encoder_width-base_tokenizer.embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True)

        # self.all_token_counts = torch.arange(0, 256+32, 32)[1:]
        # self.all_token_counts = torch.cat((torch.arange(1, 16, 1), torch.arange(0, 256+128, 16)[1:]), dim=0)
        self.all_token_counts = torch.arange(0, 256+128, 16)[1:] ## original
        # print("all_token_counts: ", self.all_token_counts)
        
        # we discretize the reconstruction loss used for conditioning KARL encoder.
        # TODO: might not be important to discretize.
        self.rec_losses = torch.tensor([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.38])
        self.rec_loss_embeddings = nn.Parameter(self.rec_losses[:, None].repeat(1, encoder_width), requires_grad=True)

        self.logged_min_loss_values_prob = torch.zeros_like(self.rec_losses)
        self.logged_min_loss_values_prob[(self.rec_losses <= 0.04)] = 1.0
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

        if self.train_stage=="full_finetuning":
            # TODO: Ablate the requirement of different discriminators for different recurrent rollout iterations.
            # Intuition is at different rollout iteration .....
            from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
            self.gan_losses = nn.ModuleList([VQLPIPSWithDiscriminator(
                disc_conditional= False, disc_in_channels= 3, 
                disc_start= 0, disc_weight= 0.2, codebook_weight= 1.0, # perceptual_weight=0.0
            ) for _ in range(max_rollout_iters)])
        
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
    
    def preprocess_encoder(self, grid_shape_2d):
        sampled_positional_embeddings = F.interpolate(self.encoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.encoder_positional_embedding.shape[0], -1).permute(0,2,1)
        return sampled_positional_embeddings
    
    def preprocess_decoder(self, img_tokens, grid_shape_2d):
        mask_tokens = self.decoder_mask_token.repeat(img_tokens.shape[0], img_tokens.shape[1], 1).to(img_tokens.dtype)
        sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
        mask_tokens = mask_tokens + sampled_positional_embeddings
        return mask_tokens


    def reconstruct_images(self, logits, code, reconstruction_shape):
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
        encoder_sampled_positional_embeddings = self.preprocess_encoder(grid_shape_2d)
        return masked_2d_tokens, encoder_sampled_positional_embeddings


    def sample_token_counts(self, batch_size, sampled_token_counts=None, sampled_code_losses=None):
        tokens_to_sample = 1
        if self.training:
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
            token_counts = sampled_token_counts
        
        if sampled_code_losses is None:
            indices = torch.multinomial(self.logged_min_loss_values_prob, batch_size*len(token_counts), replacement=True)
            predefined_reconstruction_loss = self.rec_losses[indices].reshape(batch_size, len(token_counts), 1)
            rec_losses = predefined_reconstruction_loss
            code_losses = rec_losses
        else:
            if isinstance(sampled_code_losses, float):
                code_losses = torch.zeros((batch_size, len(token_counts), 1)) + sampled_code_losses
            else:
                code_losses = sampled_code_losses
        
        assert(code_losses.shape[-1]==1)
        return token_counts, code_losses
    

    def get_discretized_loss_and_embedding(self, rec_loss_tensor, token_add_count=None):
        def discretize(loss_tensor, bin_values, bin_embeddings, token_add_count):
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

            if token_add_count is not None and token_add_count[0]==0:
                if random.random()>0.55:
                    rand = torch.rand_like(bin_indices, dtype=torch.float)  # Random floats [0, 1)
                    bin_indices = 1 + (rand * bin_indices.float()).floor().to(dtype=bin_indices.dtype)
            elif token_add_count is not None and token_add_count[0]==256-32:
                if random.random()>0.55:
                    target = len(self.rec_losses) - 1
                    rand = torch.rand_like(bin_indices, dtype=torch.float)
                    bin_indices = bin_indices.float() + rand * (target - bin_indices.float())
                    bin_indices = torch.clamp(bin_indices, max=target).to(dtype=bin_indices.dtype).long()
                
            binned_loss = bin_values[bin_indices]               # [B, S]
            binned_embed = bin_embeddings[bin_indices]          # [B, S, D]

            return binned_loss, binned_embed

        binned_rec_loss, binned_rec_embed = discretize(
            rec_loss_tensor, self.rec_losses, self.rec_loss_embeddings, token_add_count
        )
        return binned_rec_loss[...,None], binned_rec_embed[:,:,None]

    def encode(self, imgs, input_token_budget=[256], desired_reconstruction_quality=0.05):

        _, all_logs = self.forward(imgs, sampled_token_counts=input_token_budget, sampled_code_losses=desired_reconstruction_quality)
        all_embeddings = []
        all_reconstructions = []
        for iter, iter_logs_dict in enumerate(all_logs):
            for key in iter_logs_dict.keys():
                if "reconstructed_imgs" in key:
                    all_reconstructions.append(iter_logs_dict[key])
    
        return all_embeddings, all_reconstructions, all_logs

    def forward_single_image(self, imgs, sampled_token_counts=None, sampled_code_losses=None):
        all_logs = []
        sampled_token_counts, sampled_code_losses = self.sample_token_counts(batch_size=imgs.shape[0], sampled_token_counts=sampled_token_counts, sampled_code_losses=sampled_code_losses)
        sampled_code_losses, sampled_code_loss_embeddings = self.get_discretized_loss_and_embedding(sampled_code_losses[...,0])

        img_tokens, gt_indices, grid_shape_2d = self.get_2d_tokens(imgs)
        total_loss, logs, code_losses = self.forward_call(imgs,
            img_tokens, gt_indices, grid_shape_2d, is_latent_halting=False,
            sampled_token_counts=sampled_token_counts, sampled_code_loss_embeddings=sampled_code_loss_embeddings,
        )
        all_logs += logs

        if self.training:
            sampled_token_counts_2, sampled_code_losses = self.sample_token_counts(batch_size=imgs.shape[0], sampled_token_counts=sampled_token_counts, sampled_code_losses=torch.stack(code_losses, dim=1))
            # sampled_token_count_delta = [max(0, sampled_token_counts_2[idx]-max(32, sampled_token_counts[idx])) for idx in range(len(sampled_token_counts_2))]
            sampled_token_count_delta = [max(0, sampled_token_counts_2[idx]-sampled_token_counts[idx]) for idx in range(len(sampled_token_counts_2))]
            sampled_code_losses, sampled_code_loss_embeddings = self.get_discretized_loss_and_embedding(sampled_code_losses[...,0])
            total_loss, logs, code_losses = self.forward_call(imgs,
                img_tokens, gt_indices, grid_shape_2d, is_latent_halting=True,
                sampled_token_counts=sampled_token_counts_2,
                sampled_code_loss_embeddings=sampled_code_loss_embeddings,
                sampled_token_count_delta=sampled_token_count_delta,
                total_loss=total_loss
            )
            all_logs += logs
        return total_loss, all_logs


    def forward(self, imgs, epoch=None, gan_optimizer_idx=None, gan_loss_weight=None, sampled_token_counts=None, sampled_code_losses=None):
        self.epoch = epoch
        self.gan_optimizer_idx = gan_optimizer_idx
        self.gan_loss_weight = gan_loss_weight
        total_loss, all_logs = self.forward_single_image(imgs, sampled_token_counts=sampled_token_counts, sampled_code_losses=sampled_code_losses)
        return total_loss, all_logs


    def forward_call(self, imgs, img_tokens, gt_indices, grid_shape_2d, is_latent_halting,
            sampled_token_counts, sampled_code_loss_embeddings, sampled_token_count_delta=None, total_loss=0.):
        
        all_logs = []
        masked_2d_tokens, encoder_sampled_positional_embeddings = self.get_positional_embeddings(img_tokens, grid_shape_2d)
        total_loss, logs, code_losses = self.forward_executer(imgs,
            img_tokens, encoder_sampled_positional_embeddings, masked_2d_tokens, gt_indices, 
            is_latent_halting=is_latent_halting, total_loss=total_loss,
            sampled_token_counts=sampled_token_counts, sampled_code_loss_embeddings=sampled_code_loss_embeddings, 
            sampled_token_count_delta=sampled_token_count_delta, grid_shape_2d=grid_shape_2d
        )
        all_logs += logs
        return total_loss, all_logs, code_losses
        
    def prepare_decoder_latents(self, latent_embeddings, num_img_tokens, is_latent_halting, halting_prob=0.75):
        latent_tokens_halt_logits = self.halting_mlp(self.encoder_ln_post_halt(latent_embeddings))
        latent_tokens_halt_prob = F.sigmoid(latent_tokens_halt_logits)
        latent_tokens_factorized = self.pre_quantizer_mlp(self.encoder_ln_post(latent_embeddings))
        
        attn_mask = None
        if self.training and not is_latent_halting:
            return latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, attn_mask

        bs = latent_embeddings.shape[0]
        device = latent_embeddings.get_device()
        num_latent_tokens = latent_tokens_factorized.shape[1]
        num_all_tokens = num_img_tokens + num_latent_tokens
        
        img_token_mask = torch.zeros(bs, num_img_tokens, dtype=torch.bool, device=device)
        latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>halting_prob)
        combined_mask = torch.cat([img_token_mask, latent_tokens_halt_prob_mask.bool()[...,0]], dim=1)  # shape: (bs, num_all_tokens)
        attn_mask = combined_mask.unsqueeze(2) | combined_mask.unsqueeze(1)  # shape: (bs, num_all_tokens, num_all_tokens)
        diag_idx = torch.arange(num_all_tokens, device=device)
        attn_mask[:, diag_idx, diag_idx] = False
        attn_mask = attn_mask[:,1:,1:] #decoder doesn't have sampled loss value as condition which encoder (and hence corresponding attn_mask) has.
        
        return latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, attn_mask
        

    def forward_executer(self, imgs,
            img_tokens, encoder_sampled_positional_embeddings, masked_2d_tokens, gt_indices, is_latent_halting, total_loss,
            sampled_token_counts, sampled_code_loss_embeddings, sampled_token_count_delta, grid_shape_2d):

        all_iter_code_loss = []
        all_logs = []
        for iter in range(len(sampled_token_counts)):
            pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:sampled_token_counts[iter]]

            ## Latent-Distillation Encoder
            x = img_tokens + encoder_sampled_positional_embeddings
            x = torch.cat([x, (self.latent_tokens[:sampled_token_counts[iter]])[None].repeat(x.shape[0], 1, 1)], dim=1)
            x = torch.cat((sampled_code_loss_embeddings[:,iter,0:1].to(x.get_device()), x), dim=1)
            x = self.encoder_ln_pre(x)
            x = self.encoder(x, attn_mask=None, adaln_timestep_cond=None)

            ## Predict Token Halting 
            latent_tokens_factorized, latent_tokens_halt_prob, latent_tokens_halt_logits, decoder_attn_mask = \
                self.prepare_decoder_latents(x[:, 1+img_tokens.shape[1]:], num_img_tokens=1+img_tokens.shape[1], is_latent_halting=is_latent_halting)
                
            ## Latent Token Quantization
            iter_logs_dict = {}
            if self.quantize_latent:
                latent_tokens_quantized, quant_result_dict = self.quantize(latent_tokens_factorized, is_quantize=True)
            else:
                latent_tokens_quantized = latent_tokens_factorized
            
            ## Latent-Distillation Decoder
            decoded_latent_1D_tokens = self.decoder_embed(latent_tokens_quantized)
            decoded_logits = self.decoder(decoded_latent_1D_tokens, masked_2d_tokens, pos_embed_indices, attn_mask=decoder_attn_mask) 
            decoded_logits_softmax = torch.nn.functional.softmax(decoded_logits, dim=-1)
            decoded_code = torch.einsum('nlc,cd->nld', decoded_logits_softmax, self.base_tokenizer.vqgan.quantize.embedding.weight.data)            
            
            ## Loss Computation -- KC inspired Halting Loss
            latent_tokens_halt_prob_mask = (latent_tokens_halt_prob>0.75)*1.
            if self.training:    
                if is_latent_halting:
                    masked_token_count = sampled_token_count_delta[iter]
                    if masked_token_count==0:
                        latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits, torch.zeros_like(latent_tokens_halt_prob))
                        latent_token_halt_loss = None
                    else:
                        latent_token_halt_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,-masked_token_count:], torch.ones_like(latent_tokens_halt_prob[:,-masked_token_count:]))
                        latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits[:,:-masked_token_count], torch.zeros_like(latent_tokens_halt_prob[:,:-masked_token_count]))
                        total_loss = total_loss + latent_token_halt_loss
                        iter_logs_dict.update({
                            "latent_token_halt_loss_bce_{}".format(latent_tokens_factorized.shape[1]): latent_token_halt_loss.item(),
                        })
                    
                    total_loss = total_loss + latent_token_active_loss
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}".format(latent_tokens_factorized.shape[1]): latent_token_active_loss.item(),
                    })
                else:
                    masked_token_count = 256 - latent_tokens_factorized.shape[1]
                    latent_token_active_loss = F.binary_cross_entropy_with_logits(latent_tokens_halt_logits, torch.zeros_like(latent_tokens_halt_prob))
                    total_loss = total_loss + 1. * latent_token_active_loss
                    iter_logs_dict.update({
                        "latent_token_active_loss_bce_{}_0".format(latent_tokens_factorized.shape[1]): latent_token_active_loss.item(),
                    })

                ## Loss Computation -- Reconstruction Losses
                if self.train_stage == "latent_distillation_pretrain":
                    iter_nll_loss, iter_code_loss = self.forward_loss(gt_indices, decoded_logits, decoded_code)
                    
                    total_loss = total_loss + (iter_nll_loss + 1. * iter_code_loss)
                    iter_logs_dict.update({
                        "nll_loss_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): iter_nll_loss.item(),
                        "code_loss_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): iter_code_loss.item(),
                    })

                    if not is_latent_halting:
                        with torch.no_grad():
                            reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                            iter_rec_loss = torch.abs(imgs.contiguous() - reconstructed_imgs.contiguous()).reshape(decoded_code.shape[0], -1).mean(dim=-1)
                        

                elif self.train_stage == "full_finetuning":
                    reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                    gan_loss, logs_dict, iter_rec_loss = self.forward_gan_losses(
                        imgs, reconstructed_imgs, is_latent_halting, 
                        optimizer_idx=self.gan_optimizer_idx, latent_token_count=max(0,256-masked_token_count), 
                        discriminator_loss_weight=self.gan_loss_weight
                    )

                    total_loss = total_loss + gan_loss
                    iter_logs_dict.update(logs_dict)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}_{}".format(latent_tokens_factorized.shape[1], is_latent_halting*1): reconstructed_imgs,
                    })
                    
                ## Loss Computation -- Quantization Loss
                if self.quantize_latent: 
                    total_loss = total_loss + (1. * quant_result_dict['quantizer_loss'])
                    iter_logs_dict.update({
                        "quantization_loss_{}".format(latent_tokens_factorized.shape[1]): quant_result_dict['quantizer_loss'].item(),
                    })

                ## For Estimate Image Complexity Phase, rather than only sampling epsilon=0 as lossless compression, 
                ## we decay the probablity of sampling higher epsilons in this phase over training iterations (max epsilon=0.04). 
                if is_latent_halting is False: 
                    all_iter_code_loss.append(iter_rec_loss[...,None])
                    decay = 0.99
                    self.logged_min_loss_values_prob[(self.rec_losses >= iter_rec_loss.min().item())] *= decay
                    self.logged_min_loss_values_prob /= self.logged_min_loss_values_prob.sum()

            
            
            if  not self.training:
                if "reconstructed_imgs_{}".format(latent_tokens_factorized.shape[1]) not in iter_logs_dict:
                    reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}".format(latent_tokens_factorized.shape[1]): reconstructed_imgs,
                    })
                iter_rec_loss = torch.abs(imgs.contiguous() - reconstructed_imgs.contiguous()).reshape(decoded_code.shape[0], -1).mean(dim=-1)
                print("input tokens: {} | approx. kolmogorov complexity: {} | reconstruction l1 loss: {}".format(latent_tokens_factorized.shape[1], (latent_tokens_factorized.shape[1] - latent_tokens_halt_prob_mask[...,0].sum(dim=-1)).mean(), iter_rec_loss.item()))
                
            
            all_logs.append(iter_logs_dict)

        if not self.training: return total_loss, all_logs, all_iter_code_loss
        return total_loss, all_logs, all_iter_code_loss
    

    def forward_loss(self, gt_indices, decoded_logits, decoded_code):
        bsz, seq_len = gt_indices.size()
        assert(bsz==decoded_code.shape[0])
        assert(seq_len==decoded_code.shape[1])
        
        nll_loss_pixel, _ = self.criterion(decoded_logits[:, :, :self.base_tokenizer.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        nll_loss_pixel = nll_loss_pixel.reshape(bsz, seq_len)

        vqgan_embedding_shape = self.base_tokenizer.vqgan.quantize.embedding.weight.data.shape[-1]
        gt_code = torch.gather(self.base_tokenizer.vqgan.quantize.embedding.weight.data, dim=0, index=gt_indices.reshape(bsz*seq_len)[...,None].repeat(1, vqgan_embedding_shape))
        gt_code = gt_code.reshape(bsz, seq_len, vqgan_embedding_shape)
        assert(gt_code.shape == decoded_code.shape)
        code_loss = (gt_code - decoded_code)**2
        
        return nll_loss_pixel.mean(), code_loss.mean()



    def get_last_layer(self):
        return self.base_tokenizer.vqgan.decoder.conv_out.weight

    def forward_gan_losses(self, imgs, reconstructed_imgs, is_latent_halting, optimizer_idx, latent_token_count, discriminator_loss_weight):
        assert(optimizer_idx is not None)
        # iter_idx = 0 # min(4, (latent_token_count // 64))
        iter_idx = max(0,min(8, (latent_token_count // 32))-1)
        if discriminator_loss_weight==0:
            global_step=-torch.inf
            self.gan_losses[iter_idx].discriminator_weight = 0.2
        else:
            global_step=torch.inf
            self.gan_losses[iter_idx].discriminator_weight = discriminator_loss_weight
        if optimizer_idx == 0:
            aeloss, log_dict_ae, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")

            iter_log_dict_ae = {}
            for key in log_dict_ae.keys():
                iter_log_dict_ae["{}_{}_{}".format(key, latent_token_count, is_latent_halting*1)] = log_dict_ae[key]

            return aeloss, iter_log_dict_ae, iter_rec_loss

        if optimizer_idx == 1:
            discloss, log_dict_disc, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")
            
            iter_log_dict_disc = {}
            for key in log_dict_disc.keys():
                iter_log_dict_disc["{}_{}".format(key, iter_idx)] = log_dict_disc[key]
            
            return discloss, iter_log_dict_disc, iter_rec_loss