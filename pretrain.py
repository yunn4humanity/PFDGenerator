# Step 1: 필요한 라이브러리 임포트 및 데이터셋 불러오기
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig, AdamW, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch

# Step 2: 데이터셋 불러오기
dataset = load_dataset("text", data_files={"train": "data/train.txt", "validation": "data/validation.txt"})

# Step 3: 모델 및 토크나이저 초기화
config = AutoConfig.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
model = GPT2LMHeadModel(config)

# Step 4: 데이터 토큰화 함수 수정
def tokenize_function(examples):
    # 데이터를 토큰화하고 텐서 형식으로 변환
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# 토큰화 및 텐서 변환
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 5: 데이터 로더 준비 (DataCollator에서 pad_to_multiple_of 설정 추가)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True, collate_fn=data_collator)
validation_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator)

# Step 6: 모델 학습
optimizer = AdamW(model.parameters(), lr=5e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):
    for batch in train_dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["input_ids"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Step 7: 모델 저장
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")
