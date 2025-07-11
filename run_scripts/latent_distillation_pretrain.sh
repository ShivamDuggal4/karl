## Single node run comamnd
torchrun --nproc_per_node=8 \
    --master_port=12345 main_pretrain.py \
    --batch_size 64 \
    --num_workers 10 \
    --model karl_small \
    --base_tokenizer vqgan \
    --quantize_latent \
    --factorize_latent \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.e-4 --weight_decay 0.05 \
    --output_dir ./output_dir/latent_distillation_pretrain/karl_small_vqgan_quantized_latents/ \
    --data_path $TRAIN_DATA_DIR