export CUDA_VISIBLE_DEVICES=1
BID=14000
EID=20000
python -m qwq_gen \
  --load_path /work/zhuotaodeng/yzj/WaveCoder-main/data/leo_evol_16k.json \
  --save_path /work/zhuotaodeng/yzj/data_generation/data_16k_qwq/qwq_16k_${BID}.jsonl \
  --begin_id $BID \
  --end_id $EID \
  --model_name_or_path /data/zhuotaodeng/yzj/download_from_modelscope/Qwen/QwQ-32B-Preview \
  --is_use_vllm true \
  --do_sample false \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 4096 \
  --n_problems_per_batch 20 \
  --n_samples_per_problem 1 \
  --n_batches 1 