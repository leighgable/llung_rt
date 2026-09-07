import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset
from jinja2 import Template

# Standard fallback prompt templates
DEFAULT_TEACHER_TEMPLATE = """{% if system %}<|im_start|>system
{{ system }}<|im_end|>
{% endif %}<|im_start|>user
Query: {{ query }}
Demonstration Context/Answer: {{ answer }}<|im_end|>
<|im_start|>assistant
"""

DEFAULT_STUDENT_TEMPLATE = """{% if system %}<|im_start|>system
{{ system }}<|im_end|>
{% endif %}<|im_start|>user
{{ query }}<|im_end|>
<|im_start|>assistant
"""

def extract_query_and_answer(row: Dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detects common HF dataset schemas (ShareGPT, OpenAI Messages, Query/Answer).
    Returns: (system_prompt, query, expert_answer)
    """
    # Pattern 1: OpenAI / ShareGPT 'messages' column (e.g., Dolci-Instruct-SFT-Tool-Use)
    if "messages" in row:
        messages = row["messages"]
        system = None
        user_msg = None
        assistant_msg = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user" and user_msg is None:
                user_msg = content
            elif role == "assistant" and assistant_msg is None:
                assistant_msg = content
                
        return system, user_msg, assistant_msg

    # Pattern 2: Prompt / Completion or Question / Answer
    query = row.get("query") or row.get("prompt") or row.get("question") or row.get("input")
    answer = row.get("answer") or row.get("completion") or row.get("response") or row.get("output")
    system = row.get("system") or row.get("system_prompt")
    
    return system, query, answer

def stream_distillation_dataset(
    dataset_name: str,
    split: str = "train",
    config_name: Optional[str] = None,
    teacher_template_str: str = DEFAULT_TEACHER_TEMPLATE,
    student_template_str: str = DEFAULT_STUDENT_TEMPLATE,
) -> Iterator[Dict[str, str]]:
    """
    Streams a Hugging Face dataset and yields formatted Teacher and Student prompt pairs.
    """
    # Streaming = True ensures zero disk downloading upfront
    ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True)
    
    teacher_tmpl = Template(teacher_template_str)
    student_tmpl = Template(student_template_str)

    for row in ds:
        system, query, answer = extract_query_and_answer(row)
        
        # Skip invalid/corrupted samples missing either question or target answer
        if not query or not answer:
            continue
            
        render_vars = {"system": system, "query": query, "answer": answer}
        
        teacher_prompt = teacher_tmpl.render(**render_vars)
        student_prompt = student_tmpl.render(**render_vars)
        
        yield {
            "teacher_prompt": teacher_prompt,
            "student_prompt": student_prompt,
            "target_completion": answer
        }

def load_model_and_tokenizer(
    model_name: str,
    device: str,
):
    typer.echo(f" Loading Model ' {model_name} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    return model, tokenizer
