# app/llm_pseudocode.py
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an algorithm reasoning engine.
Convert any natural language description of a process or algorithm into
a structured pseudocode JSON representation.

Your output must be ONLY JSON with these fields:
- metadata: { domain (string), title (string, optional) }
- entities: list of objects with { id, type, (optional) shape or attributes }
- operations: ordered list of { step (int), action (string) }

Domain inference rules:
- If the text involves CNNs, kernels, convolution, padding, etc → domain="cnn_param"
- If sorting, comparisons, arrays → domain="sorting"
- If queues, caches, FIFO/LRU → domain="cache"
- If graphs, nodes, edges → domain="graph"
- Otherwise, domain="generic"

Be concise, but ensure every operation step is explicit and sequential."""

def build_prompt_pseudocode(user_text: str) -> str:
    return f"""
Text to convert:
{user_text}

Output JSON strictly matching the schema described above.
""".strip()

def call_llm_pseudocode_ir(user_text: str, temperature: float = 0.0):
    prompt = build_prompt_pseudocode(user_text)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    result = json.loads(resp.choices[0].message.content)
    return result
