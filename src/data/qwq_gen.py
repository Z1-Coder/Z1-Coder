import torch
import json
import itertools
from pathlib import Path
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
from typing import Any, Literal, Dict, Mapping, Iterable, Sequence, TypeVar, cast
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)

PROMPT_WRAPPER = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}

@@ Response
"""


_T = TypeVar("_T")
CUDA_NUM=torch.cuda.device_count()

@dataclass(frozen=True)
class Args:
    begin_id: int
    end_id: int
    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int

    max_new_tokens: int
    top_p: float
    max_new_tokens: int
    temperature: float
    do_sample: bool

    load_path: str
    save_path: str
    model_name_or_path: str | None = None
    use_flash_attention: bool = False
    is_use_vllm: bool = True

@dataclass(frozen=True)
class ModelContext:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer = None
    
    def complete(self, config: SamplingParams|GenerationConfig, prompts: list[str]) -> Dict:
        if self.tokenizer and isinstance(config, GenerationConfig):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_len = input_ids['input_ids'].shape[-1]
            input_ids = input_ids.to(self.model.device)
            output_ids = self.model.generate(**input_ids, generation_config=config)
            output_ids = output_ids[:, input_len:]
            output_strings = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        elif isinstance(config, SamplingParams):
            vllm_outputs = self.model.generate(prompts, config)
            output_strings =  [
                                [t.text for t in output.outputs]
                                for output in vllm_outputs
                               ]
            # input_ids = [output.outputs[0].prompt_token_ids for output in vllm_outputs]
            # output_ids = [output.outputs[0].token_ids for output in vllm_outputs]
        else:
            raise ValueError("Please check the generation configuration.")
        return dict(
            # raw_inputs=input_ids,
            # raw_outputs=output_ids,
            decoded_outputs=output_strings,
        )

def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))

def main():
    parser = HfArgumentParser(Args)
    args = cast(
        Args,
        parser.parse_args_into_dataclasses(),
    )[0]

    with open(args.load_path) as f:
        problems = json.load(f)

    problems = problems[args.begin_id:args.end_id]

    other_kwargs: dict = {}
    other_kwargs["device_map"] = "auto"
    if args.use_flash_attention:
        other_kwargs["use_flash_attention_2"] = True
    if args.is_use_vllm:
        generation_config = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p if args.do_sample else 1.0,
            n=args.n_samples_per_problem,
            max_tokens=args.max_new_tokens,
            )
        model = LLM(
            model=args.model_name_or_path, 
            max_model_len=8192, 
            download_dir='/data/zhuotaodeng/pretrained_models/',
            tensor_parallel_size=CUDA_NUM,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
            )   
        tokenizer = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            cache_dir='/data/zhuotaodeng/pretrained_models/', 
            trust_remote_code=True,
            **other_kwargs
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=True, 
            trust_remote_code=True,
            padding_side='left'
            )
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.n_samples_per_problem,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
        
    state = ModelContext(model, tokenizer)
    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches


    with open(args.save_path,'a') as f:
        for problems, batch_idx in tqdm(iter, total=n_total):
            prompts = [
                PROMPT_WRAPPER.format(
                    instruction=problem["instruction"]
                )
                for problem in problems
            ]
            print("PROMPT")
            print(prompts[-1])
            # all_prompts = prompts * args.n_samples_per_problem
            response = state.complete(generation_config, prompts)
            completions = response['decoded_outputs']
            print("COMPLETION")
            print(completions[-1])

            for id, res in enumerate(completions):
                data = {'instruction': problems[id]['instruction'], 'input': problems[id]['input'], 'gpt4o_output': problems[id]['output'], 'output': res}
                print(json.dumps(data), file=f, flush=True)


if __name__ == "__main__":
    main()