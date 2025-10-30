import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_scheduler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import json 

MODEL_NAME = 'bert-base-uncased' 
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class PersonalizedRewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        personalized_output = item["personalized_output"]
        non_personalized_output = item["non_personalized_output"]

        inputs_p = self.tokenizer(
            query,
            personalized_output,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt' 
        )

        inputs_np = self.tokenizer(
            query,
            non_personalized_output,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        return {
            'input_ids_p': inputs_p['input_ids'].squeeze(0), 
            'attention_mask_p': inputs_p['attention_mask'].squeeze(0),
            'token_type_ids_p': inputs_p['token_type_ids'].squeeze(0),
            'input_ids_np': inputs_np['input_ids'].squeeze(0),
            'attention_mask_np': inputs_np['attention_mask'].squeeze(0),
            'token_type_ids_np': inputs_np['token_type_ids'].squeeze(0),
        }

all_data=[]
import json
res_path='/LaMP_7_time/train/construct_bert_data/lamp7_bert_train.jsonl'
# res_path='/LaMP_5_time/train/construct_bert_data/lamp5_bert_train.jsonl'
# res_path='/LaMP_4_time/train/construct_bert_data/lamp4_bert_train.jsonl'

with open(res_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        all_data.append(data)
print(len(all_data))

train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = PersonalizedRewardDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
val_dataset = PersonalizedRewardDataset(val_data, tokenizer, MAX_SEQ_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RewardModel(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_train_steps = len(train_dataloader) * NUM_EPOCHS
print('total:{}'.format(total_train_steps))
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0, # 可以设置预热步数
    num_training_steps=total_train_steps,
)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids_p = batch['input_ids_p'].to(DEVICE)
        attention_mask_p = batch['attention_mask_p'].to(DEVICE)
        token_type_ids_p = batch['token_type_ids_p'].to(DEVICE)

        input_ids_np = batch['input_ids_np'].to(DEVICE)
        attention_mask_np = batch['attention_mask_np'].to(DEVICE)
        token_type_ids_np = batch['token_type_ids_np'].to(DEVICE)

        scores_p = model(input_ids_p, attention_mask_p, token_type_ids_p)
        scores_np = model(input_ids_np, attention_mask_np, token_type_ids_np)

        scores_diff = scores_p - scores_np

        loss = -F.logsigmoid(scores_diff).mean() 

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0: 
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} 训练结束，平均 Loss: {avg_train_loss:.4f}")

    model.eval()
    val_total_loss = 0
    correct_predictions = 0
    total_pairs = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids_p = batch['input_ids_p'].to(DEVICE)
            attention_mask_p = batch['attention_mask_p'].to(DEVICE)
            token_type_ids_p = batch['token_type_ids_p'].to(DEVICE)

            input_ids_np = batch['input_ids_np'].to(DEVICE)
            attention_mask_np = batch['attention_mask_np'].to(DEVICE)
            token_type_ids_np = batch['token_type_ids_np'].to(DEVICE)

            scores_p = model(input_ids_p, attention_mask_p, token_type_ids_p)
            scores_np = model(input_ids_np, attention_mask_np, token_type_ids_np)

            scores_diff = scores_p - scores_np
            val_loss = -F.logsigmoid(scores_diff).mean()
            val_total_loss += val_loss.item()

            correct_predictions += torch.sum(scores_diff > 0).item()
            total_pairs += scores_diff.size(0) 

    avg_val_loss = val_total_loss / len(val_dataloader)
    validation_accuracy = correct_predictions / total_pairs
    print(f"Epoch {epoch+1}  Loss: {avg_val_loss:.4f}, acc: {validation_accuracy:.4f}")


torch.save(model.state_dict(), "reward_model_personalized_lamp7.pth")
# torch.save(model.state_dict(), "reward_model_personalized_lamp4.pth")
# torch.save(model.state_dict(), "reward_model_personalized_lamp5.pth")





