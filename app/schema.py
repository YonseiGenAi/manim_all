# app/schema.py
from jsonschema import Draft7Validator
from typing import Dict, Any, List

JSON_IR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["components", "events"],
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    # 선택: 정적 레이아웃/메타데이터
                    "pos": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"}
                    },
                    "label": {"type": "string"},
                    "style": {"type": "object"}
                }
            }
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["t", "op"],
                "properties": {
                    "t": {"type": "number"},
                    "op": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "target": {"type": "string"},
                    "item": {"type": "string"},
                    "data": {}
                }
            }
        },
        "metadata": {"type": "object"}
    },
    "additionalProperties": True
}

def schema_errors(doc: Dict[str, Any]) -> List[str]:
    v = Draft7Validator(JSON_IR_SCHEMA)
    return [f"{e.message} at {list(e.absolute_path)}" for e in v.iter_errors(doc)]

def invariants_errors(doc: Dict[str, Any]) -> List[str]:
    """
    도메인 불변성 체크 예시:
      - events.t 오름차순
      - from/to/target 참조는 components.id 중 하나여야 함
    필요 시 알고리즘별 규칙을 더 추가하세요.
    """
    errors: List[str] = []
    comp_ids = {c["id"] for c in doc.get("components", []) if "id" in c}
    evts = doc.get("events", [])

    # 시간 오름차순
    if any(evts[i]["t"] > evts[i+1]["t"] for i in range(len(evts)-1)):
        errors.append("events.t must be non-decreasing order")

    # from/to/target 참조 유효성
    for i, e in enumerate(evts):
        for k in ("from", "to", "target"):
            if k in e and e[k] not in comp_ids:
                errors.append(f"event[{i}] references undefined '{k}': {e[k]}")

    return errors

# === seq_attention (Transformer Attention 패턴) IR 스키마 ===

ATTENTION_IR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["pattern_type", "tokens", "weights", "query_index"],
    "properties": {
        "pattern_type": {
            "type": "string",
            "const": "seq_attention",
        },
        "raw_text": {                     
            "type": "string",
        },
        "tokens": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "weights": {
            "type": "array",
            "minItems": 1,
            "items": {
                "oneOf": [
                    # 1D: [w0, w1, ..., w_{N-1}]
                    {"type": "number"},
                    # 2D: [[...], [...], ...] 도 나중에 쓸 수 있게 남겨둠
                    {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number"},
                    },
                ]
            },
        },
        "query_index": {
            "type": "integer",
            "minimum": 0,
        },
        "next_token": {                 
            "type": "object",
            "required": ["candidates", "probs"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "probs": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,     
}


ATTENTION_IR_VALIDATOR = Draft7Validator(ATTENTION_IR_SCHEMA)


def validate_attention_ir(doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    # 1) jsonschema 기반 기본 검증
    for err in ATTENTION_IR_VALIDATOR.iter_errors(doc):
        errors.append(err.message)

    tokens = doc.get("tokens", [])
    weights = doc.get("weights", [])

    # 2) weights 형태 검사
    if isinstance(weights, list) and weights:
        first = weights[0]

        # --- case 1: 2D matrix [[...], [...], ...] ---
        if isinstance(first, list):
            if len(weights) != len(tokens):
                errors.append("len(weights) must equal len(tokens) for 2D weights")

            row_len = len(first)
            if any(len(row) != row_len for row in weights):
                errors.append("all rows in weights must have the same length")

        # --- case 2: 1D row [w_0, w_1, ..., w_n-1] ---
        else:
            if len(weights) != len(tokens):
                errors.append("len(weights) must equal len(tokens) for 1D weights")

    # 3) query_index 범위 체크
    qi = doc.get("query_index")
    if isinstance(qi, int) and not (0 <= qi < len(tokens)):
        errors.append("query_index out of range")

    # 4) next_token (선택) 검증
    nt = doc.get("next_token")
    if nt is not None:
        cands = nt.get("candidates")
        probs = nt.get("probs")
        if not isinstance(cands, list) or not isinstance(probs, list):
            errors.append("next_token.candidates and probs must be lists")
        elif len(cands) != len(probs):
            errors.append("next_token.candidates and probs must have the same length")

    return errors
