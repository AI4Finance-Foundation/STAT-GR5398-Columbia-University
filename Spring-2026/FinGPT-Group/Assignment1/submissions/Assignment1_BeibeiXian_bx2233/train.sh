# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0

ds \
    ### Adjust based on your setup
    --include localhost:0 \ 
    train_lora.py \
    --run_name nasdaq-100-20231231-20241231 \ 
    --base_model llama2 \
    --dataset fingpt-forecaster-nasdaq-100-20231231-20241231-1-4-06 \
    --max_length 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --log_interval 10 \
    --warmup_ratio 0.03 \
    --scheduler constant \
    --evaluation_strategy steps \
    --ds_config config.json
