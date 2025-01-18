export CUDA_VISIBLE_DEVICES=1
MODEL_DIR=${1}
MODEL_DIR=${MODEL_DIR:-"/data/zhuotaodeng/yzj/ckpt/qwen7b_110k_qwq_cl_19k_qwq_ckpt120/checkpoint-50"}
OUTPUT_DIR=${3}
OUTPUT_DIR=${OUTPUT_DIR:-"./evaluation/qwen7b_110k_qwq_cl_19k_qwq_ckpt120_v2"}
echo "LiveCodeBench: ${MODEL_DIR}, OUPTUT_DIR: ${OUTPUT_DIR}"

python -m lcb_runner.runner.main \
    --model ZCoder \
    --model_path ${MODEL_DIR} \
    --output_name "qwen7b_110k_qwq_cl_19k_qwq_ckpt120_v2" \
    --scenario codegeneration \
    --dtype float16 \
    --evaluate \
    --tensor_parallel_size 1 \
    --output_dir ${OUTPUT_DIR}
    
# saved_eval_all_file="${OUTPUT_DIR}/log.json"
# python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-09-01

