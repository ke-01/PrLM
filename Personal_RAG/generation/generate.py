import sys

sys.path.append('.')
import argparse
import json
import os
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data.datasets import Seq2SeqDataset
from metrics.eval_metrics import LaMPEvaluation
from prompts.post_process import load_post_process_function


def get_text_template(tokenizer: AutoTokenizer, prompt):
    message = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(message,
                                         tokenize=False,
                                         add_generation_prompt=True)


parser = argparse.ArgumentParser()

parser.add_argument("--CUDA_VISIBLE_DEVICES", default='1,0')
parser.add_argument("--random_seed", type=int, default=2024)

parser.add_argument("--base_addr", default='')
parser.add_argument("--task", default="LaMP_5_time")
parser.add_argument("--input_path", default='dev/recency/bge-base-en-v1.5_5/')
parser.add_argument("--source",
                    default='bge-reranker-base/20241009-123120_rerank_5')
parser.add_argument("--file_name",
                    default='20241009-122157_user-6_20241009-120906')

parser.add_argument("--model_name",
                    default="Meta-Llama-3-8B-Instruct",
                    choices=['Meta-Llama-3-8B-Instruct', 'Qwen2-7B-Instruct'])
parser.add_argument("--begin_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=1000000)


# Generation Config
parser.add_argument("--max_new_tokens", default=768)

parser.add_argument("--cutoff_len", default=3000)



def post_process_LaMP_7(results):
    processed_results = []
    for generate_data in results:
        generate_data=generate_data.replace("{'tweet': 'generated tweet'}","")
        # {'tweet': 'generated tweet'}
        begin_index = generate_data.find("'tweet': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"tweet\": \"")
            begin_index += len("\"tweet\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'tweet': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index

        processed_predict = generate_data[begin_index:end_index]
        # processed_results.append(processed_predict)
        input_str=processed_predict
        start_marker = "</think>\n\n{'tweet'"
        start_index = input_str.find(start_marker)

        if start_index != -1:
            # 提取子字符串之后的内容
            extracted_content = input_str[start_index + len(start_marker):].strip()
            processed_predict=extracted_content

        if len(processed_predict)<=1:
            # processed_results
            pattern = r"</think>\n\n\'tweet\':\s*\'([^\']*)\'"
            match = re.search(pattern, generate_data)

            if match:
                processed_predict = match.group(1).strip()

        processed_results.append(processed_predict)
        
    return processed_results

def post_process_LaMP_4(results):
    processed_results = []
    for generate_data in results:
        generate_data=generate_data.replace("{'title': 'generated title'}","")
        begin_index = generate_data.find("'title': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"title\": \"")
            begin_index += len("\"title\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'title': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index
        processed_predict = generate_data[begin_index:end_index]
        # processed_results.append(processed_predict)
    
        input_str=processed_predict
        start_marker = "</think>\n\n{'title': '"
        start_index = input_str.find(start_marker)

        if start_index != -1:
            # 提取子字符串之后的内容
            extracted_content = input_str[start_index + len(start_marker):].strip()
            processed_predict=extracted_content
        if len(processed_predict)<=1:
            # processed_results
            pattern = r"</think>\n\n\'title\':\s*\'([^\']*)\'"
            match = re.search(pattern, generate_data)

            if match:
                processed_predict = match.group(1).strip()
        processed_results.append(processed_predict)
        
    return processed_results


if __name__ == "__main__":
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES

    opts.base_addr = os.path.join(f"{opts.model_name}_outputs/", opts.task,
                                  opts.input_path)

    opts.model_path='trained_model'

    for flag, value in opts.__dict__.items():
        print('{}: {}'.format(flag, value))

    tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
    # 设置分词器的填充方向为左侧。
    tokenizer.padding_side = "left"
    # 如果分词器没有定义填充标记，则使用结束标记（EOS）作为填充标记。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 构建输入文件路径和输出目录路径。
    opts.input_file = os.path.join(opts.base_addr, opts.source,
                                   f'{opts.file_name}.json')
    out_file_name = f"{opts.file_name}_vllm_new-{opts.max_new_tokens}"
    opts.output_dir = os.path.join(opts.base_addr, opts.source, out_file_name)


    # 每个 prompt 使用 apply_chat_template 构造，兼容 chat 模型格式
    eval_dataset = Seq2SeqDataset(opts.input_file,
                                  task=opts.task,
                                  llm_tokenizer=tokenizer,
                                  max_length=opts.cutoff_len,
                                  begin_idx=opts.begin_idx,
                                  end_idx=opts.end_idx)
    # 后处理输出文本
    post_process_fun = load_post_process_function(opts.task)
    generate_results = []
    all_scores = None

    prompts = [
        get_text_template(tokenizer, eval_dataset[i]['input'])
        for i in range(len(eval_dataset))
    ]
    llm = LLM(model=opts.model_path,
              gpu_memory_utilization=0.39,
              max_seq_len_to_capture=opts.cutoff_len,
              max_model_len=opts.cutoff_len)

    # 使用 vLLM 生成内容
    sampling_params = SamplingParams(seed=opts.random_seed,
                                     temperature=0,
                                     best_of=1,
                                     max_tokens=opts.max_new_tokens)

    model_outputs = llm.generate(prompts, sampling_params)
    model_preds = [x.outputs[0].text for x in model_outputs]
    processed_preds = post_process_fun(model_preds)
    # processed_preds = post_process_LaMP_7(model_preds)

    

    # 获取数据集中的真实标签。
    ground_truth = [
        eval_dataset[i]['output'] for i in range(len(eval_dataset))
    ]

    # 使用LaMPEvaluation类评估生成结果。计算每个预测的分数。
    eval_method = LaMPEvaluation(opts.task)
    pred_scores = eval_method.compute_metrics(processed_preds,
                                              ground_truth,
                                              avg=False)

    # 遍历每个数据项，将生成的文本、预测结果、真实标签和分数保存到字典中。
    for idx in tqdm(range(len(prompts))):
        data = eval_dataset[idx]
        save_dict = {
            "user_id": data['user_id'],
            "input": data['input'],
            "output": model_preds[idx],
            "predict": processed_preds[idx],
            "label": data['output']
        }
        scores = {k: v[idx] for k, v in pred_scores.items()}
        if all_scores is None:
            all_scores = {k: [v] for k, v in scores.items()}
        else:
            for k in all_scores.keys():
                all_scores[k].append(scores[k])
        save_dict.update(scores)
        generate_results.append(save_dict)

    opts.end_idx = opts.begin_idx + len(eval_dataset)
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    with open(os.path.join(
            opts.output_dir,
            f"predictions_{opts.begin_idx}-{opts.end_idx}.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(generate_results, file, indent=4, ensure_ascii=False)

    mean_scores = {k: float(np.mean(v)) for k, v in all_scores.items()}
    print(mean_scores)
    with open(os.path.join(opts.output_dir,
                           f"scores_{opts.begin_idx}-{opts.end_idx}.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(mean_scores, file, indent=4, ensure_ascii=False)
