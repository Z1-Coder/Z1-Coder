export CUDA_VISIBLE_DEVICES=1,2
export WANDB_DISABLED=true
set -ex
LOG_PATH=./log.txt
SAVE_PATH=./ckpt/qwen14b_stage1
mkdir -p $SAVE_PATH

torchrun --nproc_per_node=2 --master_port=20011 train/train.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B \
    --data_path  .data/stage1_evol_110k.json \
    --bf16 True \
    --tf32 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 17 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --lazy_preprocess False 
    > $LOG_PATH 2>&1
