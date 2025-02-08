import os
import json
import sys
import argparse

sys.path.append("../../../livecode_bench")
from lcb_runner.runner.parser import get_args
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)
def main():
    # TODO: we can add --continue_existing_evaluation flag to continue from existing evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_path",
        type=str,
        default = None,
        help="Generation path",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default = None,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument("--timeout", type=int, default=18, help="Timeout for evaluation")
    
    args = parser.parse_args()
    output_path = args.output_dir + '/result.json'
    benchmark, format_prompt = build_prompt_benchmark(args.scenario)

    with open(args.generation_path) as f:
        save_results = json.load(f)

    combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])

    save_eval_results = [instance.insert_output_evaluation(outputs_list, extracted_list, graded_list) for instance, (outputs_list, extracted_list), graded_list in zip(benchmark, combined_results, graded)]

    with open(output_path.replace(".json", "_eval.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
        json.dump(save_eval_results, f, indent=4)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(save_results, open(f"{args.output_dir}/generation.json", "w"), indent=4)
        json.dump(metrics, open(f"{args.output_dir}/results.json", "w"), indent=4)
        json.dump(save_eval_results, open(f"{args.output_dir}/log.json", "w"), indent=4)

if __name__== '__main__':
    main()
