## Single node run comamnd
torchrun --nproc_per_node=8 \
    --master_port=12345 main_full_finetuning.py \
    --batch_size 18 \
    --model karl_small \
    --epochs 400 \
    --warmup_epochs 20 \
    --blr 1.e-4 --weight_decay 0.05 \
    --base_tokenizer vqgan \
    --quantize_latent \
    --factorize_latent \
    --output_dir ./output_dir/full_finetuning/karl_small_vqgan_quantized_latents/ \
    --finetune ./output_dir/latent_distillation_pretrain/karl_small_vqgan_quantized_latents/checkpoint-last.pth \
    --data_path $TRAIN_DATA_DIR