import os
import sys
import csv
import json
import random
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from load_retriever import build_retriever
from load_models import build_model
from eval_metrics import get_calcu_error_bool, get_calcu_bool, resp2ans
from tqdm import tqdm

random.seed(2026)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='FinBench', type=str, help="FinBench, BizBench, KnowledgeFMATH")
parser.add_argument("--task", default='bool', type=str, help="bool, mcq, calcu")
parser.add_argument("--model", default='gpt-4o', type=str, help="gpt model or local models")
parser.add_argument("--reason_type", default='CoT', type=str, help="CoT, PoT, DA")
parser.add_argument("--sys_msg", default='On', type=str, help="On, Off")
parser.add_argument("--retri_type", default='free', type=str, help="no, free, gold")
parser.add_argument("--retriever", default='bm25', type=str, help="bm25, ada")
parser.add_argument("--top_k_retr", default=3, type=int, help="normally set to 3")
parser.add_argument("--max_token", default=1024, type=int, help="max number of output tokens")
parser.add_argument("--batch_size", default=4, type=int, help="batch size for model inference")
parser.add_argument("--first_half", action='store_true', help="whether to evaluate only the first half of the sampled questions for quick test")
parser.add_argument("--second_half", action='store_true', help="whether to evaluate only the second half of the sampled questions for quick test")
args = parser.parse_args()
arg_dict = args.__dict__

dataset_ = arg_dict['dataset']
task_ = arg_dict['task']
model_ = arg_dict['model']
reason_type = arg_dict['reason_type']
sys_msg_bool = 1 if arg_dict['sys_msg'] == 'On' else 0
retri_type = arg_dict['retri_type']
if retri_type == 'free':
    retriever_ = arg_dict['retriever']
else:
    retriever_ = ''
top_k_retr = arg_dict['top_k_retr']
max_token_ = arg_dict['max_token']
batch_size_ = arg_dict['batch_size']
project_path = os.environ["PROJECT_PATH"]

# Load Model and Retriever
model = build_model(model_)
# retriever = build_retriever(retri_type, top_k_retr)

# Load QA Data
if task_ != "finqa":
    qa_data = pd.read_csv(f'{project_path}/dataset/test_set.csv')
    qa_data = qa_data[qa_data['task'] == task_]
else:
    qa_data = pd.read_json(f'{project_path}/dataset/finqa_calcu_test.json', lines=True)
qa_data = qa_data.sample(n=100, random_state=2026).reset_index(drop=True)

if args.first_half:
    qa_data = qa_data.head(4)
elif args.second_half:
    qa_data = qa_data.tail(len(qa_data) // 2)

qa_data['model_response'] = ''
qa_data['model_answer'] = -1.0
qa_data['model_ans_bool'] = 0
qa_data['model_ans_err5_bool'] = 0
qa_data['execute_time'] = ''

# Load Prompt Template
with open(file=f'{project_path}/evaluate/prompt_template/{task_}_{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
    prompt_ques = fp.read()

if sys_msg_bool == 0:
    sys_msg = ''
else:
    with open(file=f'{project_path}/evaluate/prompt_template/Sys_msg/{reason_type}.txt', mode='r', encoding='UTF-8') as fp:
        sys_msg = fp.read()

# Save with executing date and time
time_ = datetime.now()
current_time = f"{time_.year}{time_.month}{time_.day}_{time_.hour}{time_.minute}"
save_path = f"{project_path}/results/{dataset_}/{task_}/{model_.replace('/', '_')}_{reason_type}_{retri_type}_{retriever_}_{str(top_k_retr)}_{max_token_}_{current_time}_sys{sys_msg_bool}.csv"
if not os.path.exists(f"{project_path}/results/{dataset_}/{task_}/"):
    os.makedirs(f"{project_path}/results/{dataset_}/{task_}/")
print(f"\n\nSAVE PATH: {save_path}\n")

# Evaluation Start
all_indices = qa_data.index.tolist()

for start in tqdm(range(0, len(all_indices), batch_size_), desc="Evaluating"):
    batch_indices = all_indices[start:start + batch_size_]

    batch_prompts = []
    batch_fg_paths = []

    # prepare batch inputs
    for idx in batch_indices:
        if 'finqa' not in task_:
            question_ = qa_data.loc[idx]['question']
            choi_ = qa_data.loc[idx]['choice']

            if pd.isnull(qa_data.loc[idx]['figure']) == 0:
                fg_nm = qa_data.loc[idx]['figure']
                fg_path = f"./dataset/figures/{fg_nm}"
                if '.png' not in fg_path:
                    if 'p' == fg_nm[0] or 's' == fg_nm[0]:
                        fg_path = fg_path + '.png'
                    else:
                        fg_path = fg_path + '.jpg'
            else:
                fg_path = ''
        else:
            query_ = qa_data.loc[idx]['query']
            fg_path = ''

        exce_time = datetime.now()
        qa_data.loc[idx, 'execute_time'] = str(exce_time)

        if 'finqa' in task_:
            prompt_idx = prompt_ques.format(query=query_)
        elif 'mcq' in task_:
            prompt_idx = prompt_ques.format(knowledge='', question=question_, choices=choi_)
        else:
            prompt_idx = prompt_ques.format(knowledge='', question=question_)

        batch_prompts.append(prompt_idx)
        batch_fg_paths.append(fg_path)

    # batch model call
    responses_, num_tokens_ = model.get_model_response(
        sys_msg=sys_msg,
        msg=batch_prompts,
        model_name=model_,
        image_pt=batch_fg_paths,
        sys_msg_bool=sys_msg_bool,
        max_token_=max_token_
    )

    # keep old downstream logic as much as possible
    for idx, response_, num_token_ in zip(batch_indices, responses_, num_tokens_):
        ans_ = qa_data.loc[idx]['ground_truth'] if 'finqa' not in task_ else qa_data.loc[idx]['answer']

        if reason_type == 'CoT' or reason_type == 'DA':
            model_ans = resp2ans(task_, response_)
            if model_ans == '':
                model_ans = -1
            print(f"model_ans: {model_ans}")
        elif reason_type == 'PoT':
            exec_code = response_.split("```python")[-1].split("```")[0] + "\nval_ = solution()"
            if 'scipy' in exec_code:
                exec_code = 'import scipy\n' + exec_code
            if 'math' in exec_code:
                exec_code = 'import math\n' + exec_code
            try:
                local_vars = {}
                exec(exec_code, {}, local_vars)
                model_ans = local_vars['val_']
                if type(model_ans) == str:
                    model_ans = 'Therefore, my answer is' + model_ans
                    model_ans = resp2ans(task_, model_ans)
            except Exception as e:
                print(f"Error when excuting PoT codes! {e}")
                model_ans = 0

        qa_data.loc[idx, 'model_response'] = response_

        if 'calcu' in task_ or 'finqa' in task_:
            # try:
            #     qa_data.loc[idx, 'model_answer'] = model_ans
            #     if type(ans_) == str:
            #         if '-' in ans_:
            #             ans_ = float(ans_.replace('-', '')) * (-1)
            #         else:
            #             ans_ = float(ans_)
            #     qa_data.loc[idx, 'model_ans_bool'] = get_calcu_bool(ans_, model_ans)
            #     qa_data.loc[idx, 'model_ans_err5_bool'] = get_calcu_error_bool(ans_, model_ans)
            # except:
            #     qa_data.loc[idx, 'model_answer'] = "0"
            #     qa_data.loc[idx, 'model_ans_bool'] = 0
            #     qa_data.loc[idx, 'model_ans_err5_bool'] = 0
            try:
                qa_data.loc[idx, 'model_answer'] = model_ans
                if type(ans_) == str:
                    if '-' or '–' in ans_:
                        ans_ = float(ans_.replace('-', '')) * (-1)
                    else:
                        ans_ = float(ans_)
                qa_data.loc[idx, 'model_ans_bool'] = get_calcu_bool(ans_, model_ans)
                qa_data.loc[idx, 'model_ans_err5_bool'] = get_calcu_error_bool(ans_, model_ans)
            except Exception as e:
                print(f"Error when calculating calcu metrics! {e}")
                qa_data.loc[idx, 'model_answer'] = -1
                qa_data.loc[idx, 'model_ans_bool'] = 0
                qa_data.loc[idx, 'model_ans_err5_bool'] = 0
        else:
            qa_data.loc[idx, 'model_answer'] = model_ans
            qa_data.loc[idx, 'model_ans_bool'] = get_calcu_bool(ans_, model_ans)

qa_data.to_csv(save_path)
qa_data.to_json(save_path.replace('.csv', '.json'), orient='records', lines=True)

if 'calcu' in task_ or 'finqa' in task_:
    em_bool_list = qa_data['model_ans_bool'].tolist()
    err5_bool_list = qa_data['model_ans_err5_bool'].tolist()
    em_acc = sum(em_bool_list) / len(em_bool_list)
    err5_acc = sum(err5_bool_list) / len(err5_bool_list)
    print(f"\nAcc score:\nem acc: {em_acc}\nerror 0.5% acc: {err5_acc}\n{save_path}\nFinished!\n")
else:
    result_list = qa_data['model_ans_bool'].tolist()
    acc = sum(result_list) / len(result_list)
    print(f"\n\nAcc score: {acc}\n{save_path}\nFinished!\n")