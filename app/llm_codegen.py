# app/llm_codegen.py
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a Manim code generator.
You receive a structured animation IR (entities, layout, actions) and produce valid Python code
that can be executed with Manim to visualize the described process.

Your output must be a complete, executable Python script containing:
- from manim import *
- a Scene subclass named AlgorithmScene
- use of shapes (Square, Circle, Rectangle, Text, etc.) for entities
- animations following the actions sequence (FadeIn, MoveTo, Highlight, FadeOut, etc.)
- DO NOT print or log anything.
- End with self.wait(2) to pause at the end.
"""

def build_prompt_codegen(anim_ir: dict) -> str:
    return f"""
You are a Manim expert. Convert the following structured animation IR into a **complete** Manim Scene.

IR:
{json.dumps(anim_ir, indent=2, ensure_ascii=False)}

Requirements:
1. **Must visualize every operation sequentially** — no skipping.
   - Each "step" in the IR should clearly appear on screen.
   - If the IR defines queues or caches, show items moving IN, OUT, and being REINSERTED.
   - Eviction should visually remove an element, and reinsertion should show it added back.
2. Use readable layout positions and colors provided in the IR.
3. Label each main object (e.g., "S-FIFO", "M-FIFO") above its rectangle.
4. Add subtle pauses (`self.wait(0.3)`) between major steps for clarity.
5. End with a short fade-out of all objects (to signal completion).

Output:
- Write **only Python code** that defines one Manim Scene class (e.g., `class AlgorithmScene(Scene)`).
- Do not include markdown (no ```python or ```).
- Code must be directly executable by `manim`.
"""



def call_llm_codegen(anim_ir: dict, temperature: float = 0.0):
    prompt = build_prompt_codegen(anim_ir)
    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    code = resp.choices[0].message.content

    code = code.replace("```python", "").replace("```", "").strip()
    return code
