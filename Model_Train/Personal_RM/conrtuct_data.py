data_list=[]
import re
import json
res_path='data_person_lamp7.jsonl'
with open(res_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)

data_list_0=[]
res_path='data_zero_lamp7.jsonl'
with open(res_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        data_list_0.append(data)

def get_q(input_str,cate):
    if cate==7:
        pattern = r"{'tweet': 'generated tweet'} without explanation, and use only English.\n(.*)"
    elif cate==5:
        pattern=r"Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English\.\n'abstract':\s*'([^']*)'"
    elif cate==4:
        start_string = "Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n'text':"
        end_string = " 'title':"
        aa=input_str
        start_index = aa.find(start_string) + len(start_string)
        end_index = aa.find(end_string, start_index)

        extracted_content = aa[start_index:end_index]
        return extracted_content

    match = re.search(pattern, input_str, re.DOTALL)

    if match:
        extracted_content = match.group(1).strip()
    else:
        extracted_content=""
    return extracted_content

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
    input_str=processed_results[0]
    start_marker = "</think>\n\n{'tweet'"
    start_index = input_str.find(start_marker)

    if start_index != -1:
        extracted_content = input_str[start_index + len(start_marker):].strip()
        processed_results=[extracted_content]
    if len(processed_results[0])<=1:
        pattern = r"</think>\n\n\'tweet\':\s*\'([^\']*)\'"
        match = re.search(pattern, results[0])

        if match:
            processed_results = [match.group(1).strip()]
    return processed_results

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
    
    if len(processed_results[0])>50:
        for generate_data in results:
            begin_index = generate_data.find("</think>\n\n{'title':")
            if begin_index == -1:
                begin_index = generate_data.find("\"title\": \"")
                begin_index += len("\"title\": \"")
                end_index = generate_data[begin_index:].find("\"}") + begin_index
            else:
                begin_index += len("</think>\n\n{'title':")
                end_index = generate_data[begin_index:].find("\"}") + begin_index
            processed_predict = generate_data[begin_index:end_index]
            processed_results.append(processed_predict.replace("\"}",''))
    input_str=processed_results[0]
    start_marker = "</think>\n\n{'title': '"
    start_index = input_str.find(start_marker)

    if start_index != -1:
        extracted_content = input_str[start_index + len(start_marker):].strip()
        processed_results=[extracted_content]
    if len(processed_results[0])<=1:
        pattern = r"</think>\n\n\'title\':\s*\'([^\']*)\'"
        match = re.search(pattern, results[0])

        if match:
            processed_results = [match.group(1).strip()]
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
    if len(processed_results[0])<=1:
        pattern = r"</think>\n\n\'title\':\s*\'([^\']*)\'"
        match = re.search(pattern, results[0])

        if match:
            processed_results = [match.group(1).strip()]
    return processed_results

train_data=[]
for i in range(len(data_list_0)):
    new_dict={}
    inputt=data_list_0[i]['input']

    qq=get_q(inputt,7)
    outt=data_list[i]['output'].replace("{'tweet': 'generated tweet'}","")
    outt_0=data_list_0[i]['output'].replace("{'tweet': 'generated tweet'}","")
    pos_i=post_process_LaMP_7([outt])
    neg_i=post_process_LaMP_7([outt_0])

    # qq=get_q(inputt,5)
    # outt=data_list[i]['output'].replace("{'title': 'generated title'}","")
    # outt_0=data_list_0[i]['output'].replace("{'title': 'generated title'}","")
    # pos_i=post_process_LaMP_5([outt])
    # neg_i=post_process_LaMP_5([outt_0])

    # qq=get_q(inputt,4)
    # outt=data_list[i]['output'].replace("{'title': 'generated title'}","")
    # outt_0=data_list_0[i]['output'].replace("{'title': 'generated title'}","")
    # pos_i=post_process_LaMP_4([outt])
    # neg_i=post_process_LaMP_4([outt_0])

    new_dict['query']=qq
    new_dict['personalized_output']=pos_i[0]
    new_dict['non_personalized_output']=neg_i[0]
    train_data.append(new_dict)

final_data=[]
for i in range(len(train_data)):
    if len(train_data[i]['personalized_output'])==0 or len(train_data[i]['non_personalized_output'])==0:
        continue
    else:
        final_data.append(train_data[i])

import json

output_file='lamp7_bert_train_5k.jsonl'
# output_file='lamp5_bert_train_5k.jsonl'
# output_file='lamp4_bert_train_5k.jsonl'


with open(output_file, 'w', encoding='utf-8') as file:
    for item in final_data:
        json.dump(item, file, ensure_ascii=False)
        file.write('\n')