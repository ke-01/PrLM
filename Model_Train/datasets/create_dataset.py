from datasets import Dataset, DatasetDict, load_from_disk
import json, os

file_path='train.json'

# read the json file in the folder
def make_dataset(file_path):
    dataset = {"messages":[]}
    not_match_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)

    for i in range(len(data_list)):
        data =data_list[i]
        message = [{"role": "user", "content": data['input']}, {"role": "assistant", "content": str(data['label'])}]

        dataset["messages"].append(message)
    return dataset, not_match_ids

train_dataset, train_not_match_ids = make_dataset(file_path)
train_dataset = Dataset.from_dict(train_dataset)
test_dataset, test_not_match_ids = make_dataset(file_path)
test_dataset = Dataset.from_dict(test_dataset)

print('train not match ids:', len(train_not_match_ids), train_not_match_ids)
print('test not match ids:', len(test_not_match_ids), test_not_match_ids)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset.save_to_disk("datasets/file")




