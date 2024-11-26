import sys
import os

# SFILES2 폴더가 같은 디렉토리에 있으므로 현재 디렉토리 경로만 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from SFILES2.Flowsheet_Class.flowsheet import Flowsheet

# 저장된 모델과 토크나이저 불러오기
model = GPT2LMHeadModel.from_pretrained("path_to_save_model")
tokenizer = GPT2Tokenizer.from_pretrained("path_to_save_tokenizer")

# 모델을 평가 모드로 전환
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# SFILES 2.0의 단위 조작 이름을 포함한 리스트
sfiles_units = ["abs", "blwr", "centr", "comp", "cond", "cycl", "dist", "egclean", "expand", 
                "extr", "flash", "gfil", "hcycl", "hex", "lfil", "mix", "orif", "pipe", 
                "pp", "prod", "r", "raw", "reb", "rect", "scrub", "sep", "splt", "strip", 
                "v", "X"]

# 테스트할 프롬프트 입력
prompt = "(raw)(hex)(r)[(comp)(prod)]" # input("Enter a prompt: ")

# 입력 텍스트를 토큰화
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=200,  # 적절한 최대 길이 설정으로 속도 개선
        min_length=50,  # 최소 길이 설정
        num_return_sequences=3,  # 생성할 문장 수
        no_repeat_ngram_size=2,  # 반복 방지
        num_beams=3,  # Beam Search를 위해 num_beams를 2 이상으로 설정
        do_sample=True,  # Sampling 활성화
        temperature=0.95,  # 온도 설정
        top_k=100,  # Top-k 설정
        top_p=0.98,  # Top-p 설정
    )

def clean_sfiles_syntax(text):

    def replace_with_zero(match):
        # 매칭된 문자열 길이만큼 '0'으로 채움
        return '0' * len(match.group())

    def find_innermost_brackets(text):
        # 가장 안쪽의 [] 쌍을 찾음
        pattern = r'\[[^\[\]]*\]'
        match = re.search(pattern, text)
        return match

    def replace_number_pairs(text):
        result = list(text)
        for i in range(len(result)):
            if result[i] == '<' and i+1 < len(result) and result[i+1].isdigit():
                num = result[i+1]
                j = i + 2
                while j < len(result):
                    if result[j] == num:
                        result[i:i+2] = '0' * 2
                        result[j] = '0'
                        break
                    j += 1
        
        for i in range(len(result), 0):
            if result[i].isdigit():
                num = result[i]
                j = i - 1
                while j >= 0:
                    if result[j] == '<' and result[j+1] == num:
                        result[j:j+2] = '0' * 2
                        result[i] = '0'
                        break
                    j -= 1

        return result

    def check_special_pairs(text):
        stack = []
        result = list(text)
        
        i = 0
        while i < len(text):
            if text[i:i+3] == '<&|':
                stack.append(('special', i))
                i += 3
            elif text[i] == '&' and i > 0 and text[i-1] != '<':
                stack.append(('&', i))
                i += 1
            elif text[i] == '|':
                if stack and (stack[-1][0] == 'special' or stack[-1][0] == '&'):
                    start_idx = stack.pop()[1]
                    # 짝이 맞는 경우 0으로 변환
                    if stack[-1][0] == 'special':
                        result[start_idx:start_idx+3] = '000'  # <&|를 000으로
                    else:
                        result[start_idx] = '0'  # &를 0으로
                    result[i] = '0'  # |를 0으로
                i += 1
            else:
                i += 1
        
        return ''.join(result)

    pattern = r'\([a-zA-Z0-9_-]+\)'
    cleaned_text = re.sub(pattern, replace_with_zero, text)
    while True:
        match = find_innermost_brackets(cleaned_text)
        if not match:  # 더 이상 [] 쌍이 없으면 종료
            break
        # 찾은 [] 쌍을 같은 길이의 0으로 대체
        matched = re.sub(r'[\[\]]', '0', cleaned_text[match.start():match.end()+1])
        cleaned_text = cleaned_text[:match.start()] + matched + cleaned_text[match.end() + 1:]
    
    cleaned_text = replace_number_pairs(cleaned_text)
    cleaned_text = check_special_pairs(cleaned_text)

    final_text = ''.join(c for i, c in enumerate(text) if cleaned_text[i] == '0')
    
    return final_text

# 생성된 텍스트 디코딩

for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    # print("Generated Text:\n", generated_text)
    validated_text = clean_sfiles_syntax(generated_text) + "(prod)"
    print("Validated Text:\n", validated_text)

    flowsheet=Flowsheet()
    flowsheet.create_from_sfiles(validated_text, overwrite_nx=True)
    flowsheet.visualize_flowsheet(table=False, pfd_path='plots/flowsheet'+str(i), plot_with_stream_labels=False)

