# saved_eval_all_file="/data/zhuotaodeng/yzj/Qwen2.5-Coder-main/qwencoder-eval/instruct/livecode_bench/evaluation/qwen15b_ins/log.json"
OUTPUT_DIR="./evaluation/qwen7b_110k_qwq_cl_19k_qwq_ckpt120_v2"
mkdir -p ${OUTPUT_DIR}
python -m lcb_runner.runner.eval_only  \
    --generation_path "/data/zhuotaodeng/Z1-Coder/src/eval/LiveCodeBench/output/qwen7b_110k_qwq_cl_19k_qwq_ckpt120_v2/Scenario.codegeneration_10_0.2.json" \
    --scenario codegeneration  \
    --output_dir ${OUTPUT_DIR}

saved_eval_all_file="${OUTPUT_DIR}/log.json"
python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-11-01
