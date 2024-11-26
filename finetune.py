import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # 시퀀스 길이 설정
    )
    return dataset

def train_gpt2():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델과 토크나이저 로드
    model_name = "gpt2"  # 기본 GPT-2 모델 사용
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 특수 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터셋 로드
    train_dataset = load_dataset("data/finetune.txt", tokenizer)
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2는 MLM이 아닌 인과적 언어 모델링 사용
    )

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir="./gpt2_sfiles",          # 모델 저장 경로
        overwrite_output_dir=True,           # 출력 디렉토리 덮어쓰기
        num_train_epochs=3,                  # 학습 에포크 수
        per_device_train_batch_size=4,       # 배치 사이즈
        save_steps=500,                      # 모델 저장 간격
        save_total_limit=2,                  # 저장할 체크포인트 수
        logging_steps=100,                   # 로깅 간격
        learning_rate=5e-5,                  # 학습률
        warmup_steps=100,                    # 웜업 스텝
        gradient_accumulation_steps=1,       # 그래디언트 누적 스텝
    )

    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 모델 학습
    trainer.train()

    # 학습된 모델과 토크나이저 저장
    model.save_pretrained("./gpt2_sfiles/final_model")
    tokenizer.save_pretrained("./gpt2_sfiles/final_model")

if __name__ == "__main__":
    train_gpt2()