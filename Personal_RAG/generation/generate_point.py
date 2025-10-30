import sys

sys.path.append('.')
import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data.datasets import PointLabelDataset
from metrics.eval_metrics import LaMPEvaluation
from prompts.post_process import *


def get_text_template(tokenizer: AutoTokenizer, prompt):
    message = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(message,
                                         tokenize=False,
                                         add_generation_prompt=True)


parser = argparse.ArgumentParser()

parser.add_argument("--CUDA_VISIBLE_DEVICES", default='0,1')
parser.add_argument("--random_seed", type=int, default=2024)

parser.add_argument(
    "--base_addr",
    default='###outputs###/###task###/train/###input_path###/bge-base-en-v1.5_5'
)
parser.add_argument("--task", default="LaMP_2_time")
parser.add_argument("--input_path", default='recency')
parser.add_argument("--source", default='retrieval/')
parser.add_argument("--file_name",
                    default='20241009-122157_user-6_20241009-120906')

parser.add_argument("--model_name",
                    default="Meta-Llama-3-8B-Instruct",
                    choices=['Meta-Llama-3-8B-Instruct', 'Qwen2-7B-Instruct'])
parser.add_argument("--begin_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=1000000)

# Generation Config
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--cutoff_len", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=2048)

if __name__ == "__main__":
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES

    opts.base_addr = opts.base_addr.replace("###task###", opts.task)
    opts.base_addr = opts.base_addr.replace("###input_path###",
                                            opts.input_path)
    opts.base_addr = opts.base_addr.replace("###outputs###",
                                            f"{opts.model_name}_outputs")

    opts.model_path = 'LLMs/{}'.format(opts.model_name)
    for flag, value in opts.__dict__.items():
        print('{}: {}'.format(flag, value))

    tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    opts.input_path = os.path.join(opts.base_addr, opts.source,
                                   f'{opts.file_name}.json')
    out_file_name = f"point_{opts.file_name}_vllm_new-{opts.max_new_tokens}"

    opts.output_dir = os.path.join(opts.base_addr, opts.source, out_file_name)

    point_dataset = PointLabelDataset(
        opts.input_path,
        task=opts.task,
        use_date=opts.input_path.endswith('date'),
        llm_tokenizer=tokenizer,
        max_length=opts.cutoff_len,
        begin_idx=opts.begin_idx,
        end_idx=opts.end_idx)

    post_process_fun = load_post_process_function(opts.task)
    generate_results = []
    prompts = [
        get_text_template(tokenizer, point_dataset[i]['input'])
        for i in range(len(point_dataset))
    ]

    llm = LLM(model=opts.model_path,
              gpu_memory_utilization=0.37,
              max_seq_len_to_capture=opts.cutoff_len,
              max_model_len=opts.cutoff_len)

    sampling_params = SamplingParams(seed=opts.random_seed,
                                     temperature=0,
                                     best_of=1,
                                     max_tokens=opts.max_new_tokens)

    model_outputs = []
    for idx in tqdm(range(0, len(prompts), opts.batch_size)):
        batch_prompts = prompts[idx:idx + opts.batch_size]
        batch_outputs = llm.generate(batch_prompts,
                                     sampling_params,
                                     use_tqdm=False)
        model_outputs.extend(batch_outputs)
    model_preds = [x.outputs[0].text for x in model_outputs]
    processed_preds = post_process_fun(model_preds)

    ground_truth = [
        point_dataset[i]['output'] for i in range(len(point_dataset))
    ]

    eval_method = LaMPEvaluation(opts.task)
    pred_scores = eval_method.compute_metrics(processed_preds,
                                              ground_truth,
                                              avg=False)

    for idx in tqdm(range(len(prompts))):
        data = point_dataset[idx]
        save_dict = {
            "user_id": data['user_id'],
            "user_sim": data['user_sim'],
            "query": data['query'],
            "doc": data['doc'],
            "input": data['input'],
            "output": model_preds[idx],
            "predict": processed_preds[idx],
            "label": data['output']
        }
        scores = {k: v[idx] for k, v in pred_scores.items()}

        save_dict.update(scores)
        generate_results.append(save_dict)

    data_idx = 0
    group_data_list = []
    for idx, lens in enumerate(point_dataset.lengths):
        group_data = generate_results[data_idx:data_idx + lens]
        data_idx += lens
        group_data_list.append({
            "user_id": point_dataset.user_list[idx],
            "data": group_data
        })

    opts.end_idx = opts.begin_idx + len(group_data_list)
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    with open(os.path.join(
            opts.output_dir,
            f"predictions_{opts.begin_idx}-{opts.end_idx}.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(group_data_list, file, indent=4, ensure_ascii=False)
