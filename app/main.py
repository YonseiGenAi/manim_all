# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel

from app.llm_pseudocode import call_llm_pseudocode_ir
from app.llm_anim_ir import call_llm_anim_ir
from app.llm_codegen import call_llm_codegen
from app.llm import call_llm_domain_ir, call_llm_attention_ir
from app.llm_domain import call_llm_detect_domain, build_sorting_trace_ir


from app.render_cnn_matrix import render_cnn_matrix
from app.render_sorting import render_sorting
from app.render_seq_attention import render_seq_attention

from app.schema import validate_attention_ir
from app.patterns import PatternType, infer_pattern_type

import tempfile, subprocess


class GenerateRequest(BaseModel):
    text: str


app = FastAPI()


@app.post("/generate")
async def generate_visualization(req: GenerateRequest):
    user_text = req.text

    # 1️⃣ 자연어 → pseudocode IR (여기서 domain을 뽑는다)
    pseudo_ir = call_llm_pseudocode_ir(user_text)
    meta = pseudo_ir.get("metadata") or {}
    domain = meta.get("domain", "generic")

    # 2️⃣ domain + IR → pattern_type 추론
    pattern_type = pseudo_ir.get("pattern_type") or infer_pattern_type(domain, pseudo_ir)

    # 3️⃣ 패턴 타입 기준 라우팅

    # --- (A) GRID: CNN / 행렬 계열 ---
    if pattern_type == PatternType.GRID:
        # CNN 같은 경우 도메인 전용 IR 한 번 더 뽑는다
        cnn_ir = call_llm_domain_ir("cnn_param", user_text)
        cfg = cnn_ir.get("ir", {}).get("params", {})

        video_path = render_cnn_matrix(
            cfg,
            out_basename=cnn_ir.get("basename", "cnn_param_demo"),
            fmt=cnn_ir.get("out_format", "mp4"),
        )
        return {
            "domain": domain,
            "pattern_type": pattern_type.value,
            "cnn_ir": cnn_ir,
            "video_path": video_path,
        }

    # --- (B) SEQUENCE: 정렬, step-by-step ---
    if pattern_type == PatternType.SEQUENCE:
        sort_trace = build_sorting_trace_ir(user_text)
        video_path = render_sorting(sort_trace)
        return {"video_path": video_path}

    # --- (C) SEQ_ATTENTION: self-attention 시각화 ---
    if pattern_type == PatternType.SEQ_ATTENTION:
        attn_ir = call_llm_attention_ir(user_text)
        errors = validate_attention_ir(attn_ir)
        if errors:
            return {
                "domain": domain,
                "pattern_type": pattern_type.value,
                "errors": errors,
            }

        video_path = render_seq_attention(attn_ir, out_basename="attn_demo")
        return {
            "domain": domain,
            "pattern_type": pattern_type.value,
            "attention_ir": attn_ir,
            "video_path": video_path,
        }

    # --- (D) FLOW: 나중에 파이프라인 애니메이션용 ---
    if pattern_type == PatternType.FLOW:
        return {
            "domain": domain,
            "pattern_type": pattern_type.value,
            "message": "flow pattern not implemented yet",
        }

    # --- (E) fallback: 기존 generic anim_ir → codegen ---
    anim_ir = call_llm_anim_ir(pseudo_ir)
    manim_code = call_llm_codegen(anim_ir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(manim_code)
        tmp_path = tmp.name

    subprocess.run(["manim", "-ql", tmp_path, "AlgorithmScene", "--format", "mp4"])

    return {
        "domain": domain,
        "pattern_type": None,
        "pseudocode_ir": pseudo_ir,
        "anim_ir": anim_ir,
        "message": "🎬 fallback generic visualization started",
    }
