# app/patterns.py
from enum import Enum
from typing import Optional


class PatternType(str, Enum):
    GRID = "grid"              # 2D 행렬 / heatmap / CNN 등
    SEQUENCE = "sequence"      # 일렬 배열, 정렬, step-by-step
    FLOW = "flow"              # 파이프라인, 블록 다이어그램
    SEQ_ATTENTION = "seq_attention"  # 토큰 시퀀스 + attention


# 임시 도메인 → 패턴 매핑 (나중에는 LLM이 pattern_type을 직접 줄 수도 있음)
DOMAIN_TO_PATTERN = {
    "cnn_param":         PatternType.GRID,
    "sorting":           PatternType.SEQUENCE,
    "bubble_sort":       PatternType.SEQUENCE,
    "selection_sort":    PatternType.SEQUENCE,
    "attention":         PatternType.SEQ_ATTENTION,
    "transformer_attn":  PatternType.SEQ_ATTENTION,
    "transformer":       PatternType.SEQ_ATTENTION,
    # 앞으로 pipeline, loss, training loop 같은 건 FLOW로 맵핑하면 됨
    "pipeline":          PatternType.FLOW,
}


def infer_pattern_type(domain: str, ir: dict | None = None) -> Optional[PatternType]:
    """
    1순위: domain → 패턴 매핑
    2순위: 나중에 ir 내용을 보고 heuristic 추가 가능
    """
    if domain in DOMAIN_TO_PATTERN:
        return DOMAIN_TO_PATTERN[domain]

    # TODO: ir 기반 heuristic (ex: "matrix", "heatmap" → GRID 등)
    return None
