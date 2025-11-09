# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.llm import call_llm_domain_ir
from app.render_cnn_matrix import render_cnn_matrix
from app.llm_pseudocode import call_llm_pseudocode_ir
from app.llm_anim_ir import call_llm_anim_ir
from app.llm_codegen import call_llm_codegen
from openai import OpenAI
import os, tempfile, subprocess
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="GenAI Visualization API")

def sanitize_input_text(text: str) -> str:
    """
    사용자가 보낸 긴 자연어 설명을 JSON으로 안전하게 변환하기 위한 전처리기.
    - 줄바꿈(\n, \r) → 공백으로 변환
    - 연속 공백 정리
    - 따옴표 이스케이프
    - 제어 문자 제거
    """
    text = text.replace("\r", " ").replace("\n", " ")   # 줄바꿈 제거
    text = re.sub(r"\s+", " ", text)                    # 연속 공백 1개로 축소
    text = text.replace('"', '\\"')                     # 큰따옴표 이스케이프
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)         # 제어문자 제거
    return text.strip()

# (1) 공통 요청 스키마
class ParseIRRequest(BaseModel):
    text: str


# (2) 도메인 자동 감지 함수
def detect_domain_via_llm(text: str) -> str:
    prompt = f"""
    너는 입력 문장이 어떤 알고리즘/AI 개념인지 분류하는 도메인 감지기야.
    가능한 도메인 목록:
    ["cnn_param", "sorting", "transformer", "diffusion", "rnn", "cache", "math"]

    - CNN 관련 (커널, stride, padding 등) → cnn_param
    - 정렬 알고리즘 (버블, 선택, 삽입, quick sort 등) → sorting
    - Transformer / attention / QKV → transformer
    - Diffusion / noise / denoising / sampling → diffusion
    - RNN / LSTM / sequence → rnn
    - 캐시, FIFO, LRU, queue → cache
    - 수학적 계산, 미분, 행렬, 확률 → math

    문장: "{text}"

    위 문장의 도메인 이름만 하나 출력해.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict domain classifier."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()


# (3) CNN 전용 파이프라인
@app.post("/parse_ir")
async def parse_ir(req: ParseIRRequest):
    text = sanitize_input_text(req.text)
    domain = detect_domain_via_llm(text)

    # CNN 도메인만 여기서 처리
    if domain != "cnn_param":
        return {"error": f"This route handles only CNN. Detected domain: {domain}"}

    ir = call_llm_domain_ir(domain, text)


    cnn_ir = ir["ir"]
    cnn_cfg = cnn_ir.get("params", {})
    basename = ir.get("basename", "cnn_forward_param")
    out_format = ir.get("out_format", "mp4")

    video_path = render_cnn_matrix(cnn_cfg, out_basename=basename, fmt=out_format)
    return {"ir": ir, "video_path": video_path}


# (4) 범용 애니메이션 파이프라인
@app.post("/generate")
async def generate_visualization(req: ParseIRRequest):
    user_text = req.text

    # 1️⃣ 자연어 → pseudocode IR
    pseudo_ir = call_llm_pseudocode_ir(user_text)

    # 2️⃣ pseudocode → structured animation IR
    anim_ir = call_llm_anim_ir(pseudo_ir)

    # 3️⃣ animation IR → Manim 코드 생성
    manim_code = call_llm_codegen(anim_ir)

    # 4️⃣ 코드 저장 + 렌더링
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(manim_code)
        tmp_path = tmp.name

    subprocess.run(["manim", "-ql", tmp_path, "AlgorithmScene", "--format", "mp4"])

    return {
        "pseudocode_ir": pseudo_ir,
        "anim_ir": anim_ir,
        "message": "🎬 Visualization generation started successfully!"
    }
