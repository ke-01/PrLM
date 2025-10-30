"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from tqdm import tqdm
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
import numpy as np

from .utils import is_e2b_available


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_scheduler
import torch.nn.functional as F

MODEL_NAME = 'bert-base-uncased' # 或者其他BERT中文模型，如 bert-base-chinese
MAX_SEQ_LENGTH = 512 # 根据你的query和output长度调整
DEVICE = torch.device("cuda:1")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.score_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              return_dict=True)

        pooled_output = outputs.pooler_output
        score = self.score_head(pooled_output)
        
        return score

def predict_score(model, tokenizer, query, output, device):
    model.eval()
    inputs = tokenizer(
        query,
        output,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(**inputs)).squeeze().item()
    return score

model = RewardModel(MODEL_NAME).to(DEVICE)
model.load_state_dict(torch.load("train_bert/reward_model_personalized_lamp5.pth"))


model.eval()

def compute_metric_bleu_rouge(preds, labels, avg=False, use_tqdm=False):
    preds = [x.lower() for x in preds]
    labels = [x.lower() for x in labels]

    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    if use_tqdm:
        iterator = tqdm(zip(preds, labels))
    else:
        iterator = zip(preds, labels)
    for pred, label in iterator:
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if len(" ".join(hypothesis).split()) == 0 or len(
                " ".join(reference).split()) == 0:
            result = {
                "rouge-1": {
                    "f": 0.0
                },
                "rouge-2": {
                    "f": 0.0
                },
                "rouge-l": {
                    "f": 0.0
                }
            }
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis),
                                      " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(v["f"])

        bleu_score = sentence_bleu(
            [list(label)],
            list(pred),
            smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(bleu_score)

    if avg:
        return {k: float(np.mean(v)) for k, v in score_dict.items()}
    else:
        return score_dict

def post_process_LaMP_4(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'title': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"title\": \"")
            begin_index += len("\"title\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'title': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index
        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_5(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'title': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"title\": \"")
            begin_index += len("\"title\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'title': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index
        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_7(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'tweet': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"tweet\": \"")
            begin_index += len("\"tweet\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'tweet': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index

        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()


def extract_boxed_number(text: str):
    """
    从文本中提取 \boxed{a} 中的数字 a，支持整数、负数和小数。
    如果匹配不到，则返回 None。
    """
    pattern = r'\\boxed\{(-?\d+(?:\.\d+)?)\}'
    match = re.search(pattern, text)
    if match:
        # 将捕获的数字转换为浮点数返回
        return float(match.group(1))
    return None

def extract_last_number(text: str):
    """
    从文本中提取最后一个数 a，支持整数、负数和小数。
    如果匹配不到，则返回 None。
    """
    pattern = r'[-+]?\d*\.\d+|\d+'
    match = re.findall(pattern, text)
    if match:
        # 将捕获的数字转换为浮点数返回
        return float(match[-1])
    return None


# allowed_pattern = r"(<answer>|<\/answer>|<think>|<\/think>|\\boxed\{\}|[\u4e00-\u9fa5\d+\-*/$:：，。；、\s]+)"
allowed_pattern = (
    r"(<answer>|<\/answer>|<think>|<\/think>|\\boxed\{\}|"
    r"[\u4e00-\u9fa5\d\+\-\*\/\$\:：，。；、=《》<>\[\]\(\)（）\{\}\.【】%\s]+)"
)
def count_invalid_characters(text):
    """
    计算文本中不属于允许字符或标记的部分的字符数量。
    返回值为非法字符的数量和非法字符构成的字符串。
    """
    # 用允许模式替换所有匹配部分为空字符串，剩下的即为非法字符
    invalid_part = re.sub(allowed_pattern, "", text)
    return len(invalid_part), invalid_part

def penalise_invalid_characters(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in contents:
        invalid_count, invalid_part = count_invalid_characters(content)
        print(invalid_count, invalid_part)
        if invalid_count > 0:
            # 添加一个负的奖励，以惩罚非法字符
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

def distill_law_format_reward(completions, **kwargs):
    # pattern = r"^.*?\n</think>\n<answer>\n.*?\\boxed\{.*?\}\n</answer>$"
    # pattern = r"^.*?\n</think>\n.*?\\boxed\{.*?\}"
    pattern = r"^.*?\n</think>.*"


    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def law_accuracy_reward(completions, solution, question, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        res = extract_boxed_number(content)
        if res is not None and abs(float(res)- float(sol)) <1.0:
            reward = 1.0
        elif res is not None:
            reward = 0.1
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards

def get_q(input_str,cate):
    # input_str=data_list[0]['input']
    if cate==7:
        pattern = r"{'tweet': 'generated tweet'} without explanation, and use only English.\n(.*)"
    elif cate==5:
        pattern=r"Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English\.\n'abstract':\s*'([^']*)'"
    elif cate==4:
        # pattern=r"Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English\.\n'text':\s*'([^']*)'"
        start_string = "Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n'text':"
        end_string = " 'title':"
        aa=input_str
        start_index = aa.find(start_string) + len(start_string)
        end_index = aa.find(end_string, start_index)

        extracted_content = aa[start_index:end_index]
        return extracted_content.replace('\'','')

    match = re.search(pattern, input_str, re.DOTALL)

    if match:
        extracted_content = match.group(1).strip()
    return extracted_content

def rag_accuracy_reward_7(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        preds=post_process_LaMP_7([content])
        ress=compute_metric_bleu_rouge(preds,[sol])
        res=ress['rouge-1'][0]+ress['rouge-2'][0]+ress['rouge-l'][0]
        rewards.append(res)

    return rewards


def rag_accuracy_reward_5(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        preds=post_process_LaMP_5([content])
        ress=compute_metric_bleu_rouge(preds,[sol])
        res=ress['rouge-1'][0]+ress['rouge-2'][0]+ress['rouge-l'][0]
        rewards.append(res)

    return rewards

def rag_person_reward_5(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    questions = kwargs.get('question', None)
    for content, sol,qq in zip(contents, solution,questions):
        sample_query=get_q(qq,5)
        content=post_process_LaMP_5([content])[0]
        score1 = predict_score(model, tokenizer, sample_query, content, DEVICE)

        rewards.append(score1)

    return rewards

def rag_person_reward_4(completions, solution,  **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    questions = kwargs.get('question', None)

    for content, sol,qq in zip(contents, solution,questions):

        sample_query=get_q(qq,4)
        content=post_process_LaMP_4([content])[0]            
        score1 = predict_score(model, tokenizer, sample_query, content, DEVICE)

        if len(content)==0:
            score1=-10
        if "generated title" in content:
            score1=-10

        rewards.append(score1)

    return rewards

def rag_person_reward_7(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    questions = kwargs.get('question', None)

    for content, sol,qq in zip(contents, solution,questions):

        sample_query=get_q(qq,7)
        content=post_process_LaMP_7([content])[0]
        score1 = predict_score(model, tokenizer, sample_query, content, DEVICE)

        rewards.append(score1)

    return rewards

def rag_accuracy_reward_4(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        preds=post_process_LaMP_4([content])
        ress=compute_metric_bleu_rouge(preds,[sol])
        res=ress['rouge-1'][0]+ress['rouge-2'][0]+ress['rouge-l'][0]
        rewards.append(res)

    return rewards

def count_boxed_tag(text:str):
    """
        计数 \\boxed\{a\} 中的数字 a，支持整数、负数和小数。
        如果匹配不到，则返回 None。
    """
    pattern = r'\\boxed\{(-?\d+(?:\.\d+)?)\}'
    match = re.search(pattern, text)
    if match and len(match)==1:
        return True
    return False

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        frac = 1/5
        if text.count("<think>\n") == 1:
            count += frac
        if text.count("\n</think>\n") == 1:
            count += frac
        if text.count("\n<answer>\n") == 1:
            count += frac
        if text.count("\n</answer>") == 1:
            count += frac
        if  count_boxed_tag :
            count += frac*3
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def law_format_reward(completions, **kwargs):
    """
    Reward function that checks if the response is in the following format:
    
    <think>
    …
    </think>
    <response>
    …\boxed{最终计算结果}
    </response>
    
    Returns a list of rewards: 1.0 if the response matches the expected format, 0.0 otherwise.
    """
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\\boxed\{.*?\}\n</answer>$"
    pattern = r"^.*?\n</think>.*"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()
            if output.strip() == case["output"].strip():
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]
    try:
        rewards = run_async_from_sync(scripts, verification_info["language"])

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async function and get the result
        rewards = loop.run_until_complete(run_async(scripts, language))
    finally:
        loop.close()

    return rewards


async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards


async def run_script(sbx, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0
