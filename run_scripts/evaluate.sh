python evaluate.py \
    --model alit_small \
    --base_tokenizer vqgan \
    --quantize_latent \
    --output_dir ./output_dir/full_finetuning/karl_small_vqgan_quantized_latents/ \
    --ckpt ./output_dir/full_finetuning/karl_small_vqgan_quantized_latents/checkpoint-last.pth \
    --data_path $TRAIN_DATA_DIR