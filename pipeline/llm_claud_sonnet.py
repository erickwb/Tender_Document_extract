from __future__ import annotations
import hashlib
from anthropic import Anthropic

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def run_claude(
    model: str,
    prompt_template: str,
    extracted_text: str,
    edital_name: str,
) -> tuple[str, str]:
    client = Anthropic(api_key="")   # lê ANTHROPIC_API_KEY do ambiente

    filled = (
        prompt_template
        .replace("{{NOME_EDITAL}}", edital_name)
        .replace("{{TEXTO_EXTRAIDO}}", extracted_text)
    )
    p_hash = hash_text(filled)

    msg = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": filled}]
    )

    return msg.content[0].text, p_hash
