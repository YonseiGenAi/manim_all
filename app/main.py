# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.llm import generate_ir_with_validation, call_llm_domain_ir
from app.render_cnn_matrix import render_cnn_matrix 
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


# ✅ (1) 입력 스키마 정의
class ParseIRRequest(BaseModel):
    text: str


# ✅ (2) LLM을 이용한 도메인 자동 분류 함수
def detect_domain_via_llm(text: str) -> str:
    prompt = f"""
    너는 주어진 문장이 어떤 알고리즘 또는 인공지능 개념을 설명하는지 판단하는 분류기야.
    가능한 도메인 목록은 다음과 같아:
    ["cnn_param", "sorting", "transformer", "diffusion", "rnn", "math"]
    
    - "cnn_param" : CNN, 합성곱 신경망, convolution, 커널, stride, padding 같은 단어가 포함되면 선택.
    - "sorting" : 버블 정렬, 선택 정렬, 삽입 정렬, quick sort 등 정렬 알고리즘이면 선택.
    - "transformer" : self-attention, query/key/value, positional encoding 관련이면 선택.
    - "diffusion" : diffusion model, stable diffusion, noise, denoising 관련이면 선택.
    - "rnn" : recurrent, sequence, lstm, gru 관련이면 선택.
    - "math" : 수학적 계산, 행렬, 미분, 확률 등 일반 수학 연산이면 선택.

    문장: "{text}"

    위 문장의 도메인만 하나 골라서 문자열 하나만 출력해.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise domain classifier."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()


# ✅ (3) 자연어 → IR 변환
@app.post("/parse_ir")
async def parse_ir(req: ParseIRRequest):
    # 1) 사용자 자연어
    text = req.text

    # 2) 지금은 CNN 도메인만 다루니 고정
    domain = "cnn_param"

    # 3) LLM 호출해서 IR(JSON) 생성
    ir = call_llm_domain_ir(domain, text)

    # 디버깅용으로 콘솔에 찍어보기
    print("=== 🧠 LLM RAW OUTPUT ===")
    print(ir)
    print("=========================")

    # 4) IR 안에서 cnn_param용 설정 꺼내기
    cnn_ir = ir["ir"]  
    cnn_cfg = cnn_ir.get("params", {})              # {"metadata": ..., "params": {...}}
    basename = ir.get("basename", "cnn_forward_param")
    out_format = ir.get("out_format", "mp4")

    # 5) 바로 영상 렌더링
    video_path = render_cnn_matrix(
        cnn_cfg,
        out_basename=basename,
        fmt=out_format,
    )

    # 6) 클라이언트에게 IR + 영상 경로 둘 다 반환
    return {
        "ir": ir,
        "video_path": video_path,
    }


