# Kolmogorov-Approximating Representation Learning (KARL)

This is the code repository of the paper: 

> [Single-pass Adaptive Image Tokenization for Minimum Program Search](https://arxiv.org/abs/2507.07995)    
> [Shivam Duggal](https://shivamduggal4.github.io/), [Sanghyun Byun](https://shbyun080.github.io/), [William T. Freeman](https://billf.mit.edu/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Phillip Isola](https://web.mit.edu/phillipi/)    
> MIT CSAIL
>
> <p><strong>Keywords:</strong> Representation Learning, Adaptive Tokenization, Compression, Algorithmic Information Theory, Kolmogorov Complexity, Upside-Down Reinforcement Learning.</p>
> <strong>AIT </strong>(Adaptive Image Tokenization) meets <strong>AIT </strong>(Algorithmic Information Theory)!!


## Table of Content
[Abstract](#Abstract)  
[Approach Overview](#Overview)  
[Setup](#Setup)  
[Datasets](#Datasets)  
[Pretrained Checkpoints](#PretrainedCheckpoints)  
[Training](#Training)  
[Evaluation](#Evaluation)  
[Citation](#Citation)  

<a name="Abstract"></a>
## Abstract

<p style="text-align: justify;">
According to Algorithmic Information Theory (AIT), intelligent representations compress data into the shortest possible program that can reconstruct its content—exhibiting low Kolmogorov Complexity (KC). In contrast, most visual representation learning systems use fixed-length representations for all inputs, ignoring variations in complexity or familiarity. Recent adaptive tokenization methods address this by allocating variable-length representations but typically require test-time search over multiple encodings to find the most predictive one. Inspired by KC principles, we propose a single-pass adaptive tokenizer, KARL, which predicts the appropriate number of tokens for an image in a single forward pass, halting once its approximate KC is reached. The token count serves as a proxy for the minimum description length. KARL matches the performance of recent adaptive tokenizers while operating in a single pass. We present scaling laws for KARL, analyzing the role of encoder / decoder size, continuous vs. discrete tokenization and more. Additionally, we offer a conceptual study drawing an analogy between Adaptive Image Tokenization and Algorithmic Information Theory, examining the predicted image complexity (KC) across axes such as structure vs. noise and in- vs. out-of-distribution familiarity – revealing alignment with human intuition. 
</p>

<div style="text-align: justify;">
</div>

<a name="Approach Overview"></a>
## Overview

![](./assets/karl_architecture.png)

<a name="Setup"></a>
## Setup

### Environment Setup
```bash
mamba env create -f environment.yaml
mamba activate kolmogorov_tokenizer
```

### Training Setup

Training the adaptive tokenizer requires pretrained checkpoints of the base 2D image tokenizers. We use VQGAN or VAE as the base tokenizers. We acknowldege [Mage](https://github.com/LTH14/mage) / [Mar](https://github.com/LTH14/mar) for releasing Imagenet-trained checkpoints of VQGAN / VAE.  Run the following to **download the pretrained base tokenizers** at `base_tokenizers/pretrained_models`

```bash
python base_tokenizers/pretrained_models/download.py
```

To use a custom base tokenizer, add the tokenizer code in [`base_tokenizers`](base_tokenizers/), corresponding pretrained checkpoint in [`base_tokenizers/pretrained_models`](base_tokenizers/pretrained_models) and a wrapper in [`modules/base_tokenizers.py`](modules/base_tokenizers.py). See `VQGANWrapper` or `LDMVAEWrapper` for reference.


<a name="Datasets"></a>
## Datasets

We mainly used ImageNet and ImageNet100 (subset of ImageNet) for training. Download [ImageNet](https://image-net.org/download) dataset and place it in  $IMAGENET_DIR. To create the ImageNet100 sybset, run the following:

```bash
python run_scripts/create_imagenet100.py --imagenet_dir $IMAGENET_DIR --imagenet100_dir datasets/imagenet100/  
set -x IMAGENET100_DIR datasets/imagenet100/
```

<a name="PretrainedCheckpoints"></a>
## Pretrained Checkpoints

Download the required checkpoint and place it at `kolmogorov_tokenizers/pretrained_models/imagenet100/` or `kolmogorov_tokenizers/pretrained_models/imagenet/`. Optinally run the following to download all the models:

```bash
python kolmogorov_tokenizers/pretrained_models/download.py
```

| Kolmogorov Tokenizer | Base Tokenizer | Dataset | Latent Quantization | Latent Factorization | Pretrained Checkpoint |
|--------------------|----------------|----------------|----------------------|----------------------|----------------------|
| karl_small         | vqgan          |  ImageNet (1K)          | $\checkmark$         | $\checkmark$         | [Download Link](https://www.dropbox.com/scl/fi/4a2zwpceavl92ij56ja3m/karl_small_vqgan_quantized_latents.pth?rlkey=lc49wt2igsdhzfrqwvl62q30z&st=j3f9up9q&dl=0)
| karl_small         | vqgan          |  ImageNet100          | $\checkmark$         | $\checkmark$         | [Download Link](https://www.dropbox.com/scl/fi/60plmx9s1jsu5axqb1frj/karl_small_vqgan_quantized_latents.pth?rlkey=v0y3fsureyfijy5m28ba4vpl7&st=haisv8qx&dl=0)
| karl_small        | vqgan          |  ImageNet100          |  $\times$         | $\checkmark$         | [Download Link](https://www.dropbox.com/scl/fi/ipvxil7ubb41wxs25du5p/karl_small_vqgan_continuous_latents.pth?rlkey=opmua15uarn2yt60yo9j7v5uo&st=bii053ux&dl=0)
| karl_small         | vae          |  ImageNet100          |  $\checkmark$         | $\checkmark$         | [Download Link](https://www.dropbox.com/scl/fi/gynbn6onb5tm1t8pmca0x/karl_small_vae_quantized_latents.pth?rlkey=93nz1j334ex57myea0qdxu7y5&st=9hjwvyc2&dl=0)
| karl_small         | vae          |  ImageNet100          |  $\times$         | $\checkmark$         | [Download Link](https://www.dropbox.com/scl/fi/v5t6trgczpaprvoow58zk/karl_small_vae_continuous_latents.pth?rlkey=0tldlcy8ir0lrectdk1o42tch&st=qzfw7g1h&dl=0)

<a name="Training"></a>
## Training

KARL is trained in two stages – `latent distillation pretrain` and `full finetuning (with gan loss)`. The only difference between these two stages is terms of parameters optimized and GAN *vs* no GAN loss –– these stages are similar to how traditional VQGAN and followups are usually trained.

- The first latent distillation pretrain stage optimizes 2D tokens $\rightarrow$ 1D tokens latent distillation encoder/decoder modules via reconstruction losses at 2D image tokens level.
- The second full finetuning stage finetunes all the network weights with losses directly on image / pixel level, namely reconstruction loss, GAN loss, perceptual loss.


### Latent Distillation Pretrain

We train the latent-distillation encoder / decoder modules in this stage, keeping image encoder / decoder fixed.

```bash
set -x TRAIN_DATA_DIR $IMAGENET100_DIR # Set to $IMAGENET_DIR, $IMAGENET100_DIR or some other dataset to change the training dataset.
bash run_scripts/latent_distillation_pretrain.sh
```
Reference guide for adaptive tokenizer arguments:  
- `--base_tokenizer` selects 2D Image Tokenizer, current options include vqgan or vae.  
- `--model` selects the adaptive tokenizer configurations. Options: `karl_small`.  
- `--quantize_latent` leads to quantization of the learned 1D tokens before decoding (this helps create compressed image representations).  
- `--factorize_latent` performs feature dimension factorization of the learned 1D tokens before quantization. If `--quantize_latent` is set True, `--factorize_latent` will be set True automatically.  
- For rest of the arguments, please refer (and directly edit) the config files at `kolmogorov_tokenizers/configs/karl_vqgan.yaml` and `kolmogorov_tokenizers/configs/karl_vae.yaml`.  
- See `--output_dir` for training logs and checkpoints.  

### Full Finetuning Pretrain

Performs full finetuning of the latent-distillation encoder / decoder and image encoder / decoder with gan losses.  

```bash
bash run_scripts/full_finetuning.sh
```

- `--finetune` loads the checkpoint trained in the previous stage (set the argument to the corresponding path accordingly).  
- See `--output_dir` for training logs and checkpoints.

To resume training from some intermediate point, remember to load weights using the tag `--resume` in any stage of the training.

<br/>

**Note:** These stages are different from the **Estimate Image Complexity** and **Learning to Tokenize Complexity** phases, which are the core contribution of KARL tokenizer and are executed in every iteration of both stages. 

![](./assets/karl_computation_graph.png)

<a name="Evaluation"></a>
## Evaluation

### Evaluate trained model on the Imagenet (same token allocation for all images)

(uploading the code soon)

### Evaluate trained model on the Imagenet (variable token allocation per image)

(uploading the code soon)

### Evaluate KARL on custom images using pre-trained checkpoints

We recommend keeping the input token budget `T` fixed at the maximum value (e.g., 256), and instead varying the desired reconstruction error $\epsilon$ to suit the task. Unlike prior approaches that fix the token count and search for the best representation, **KARL** treats $\epsilon$ as the task specification—if the downstream task can tolerate coarser reconstructions, use a higher $\epsilon$; for fine-grained reconstructions, set it lower.

```python
karl_embedding, karl_reconstruction, _ = kolmogorov_tokenizer.encode(image_tensor, input_token_budget=256, desired_reconstruction_quality=0.05) # recommend playing with desired_reconstruction_quality to meet task-requirement or dataset-requirement.
```

<div align="center">
  <img src="./assets/karl_variable_tokenization_example.png" width="75%" />
</div>


### Don't Care? Just want a short enough representation for an image 

```python
min_length_embedding, _, _ = kolmogorov_tokenizer.encode(image_tensor) # default input_token_budget=256, desired_reconstruction_quality=0.05
```
![](./assets/karl_default_reconstruction.png)

If default = 0.05 does not yield satisfactory reconstructions—for example, if important details are lost—try lowering it to 0.03. Conversely, if 0.05 already produces near-perfect reconstructions but the token count remains unnecessarily high (e.g., above 32), consider increasing it to 0.07.
Unlike other adaptive tokenizers, KARL always remain adaptive in terms of tokens utilized at test time, unless explicitly disabled by setting $\epsilon = 0$.

<a name="Citation"></a>
## Citation

If you use our code or the paper, please consider citing the following:

```
@article{duggal2024KARL,
author = {Shivam Duggal and Sanghyun Byun and William T. Freeman and Antonio Torralba and Phillip Isola},
title = {Single-pass Adaptive Image Tokenization for Minimum Program Search},
journal= {arxiv},
year = {2025}
}
```