
from openai import OpenAI
from pathlib import Path
import hashlib
import time
from collections import deque
import tiktoken


# --- Controle de TPM (tokens por minuto) ---
TPM_LIMIT = 30000
WINDOW_SECONDS = 60
_token_window = deque()  # (timestamp, tokens)


def count_tokens(text: str, model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def wait_for_tpm(tokens_needed: int) -> None:
    while True:
        now = time.time()

        # Remove entradas fora da janela
        while _token_window and (now - _token_window[0][0]) > WINDOW_SECONDS:
            _token_window.popleft()

        used = sum(t for _, t in _token_window)

        if used + tokens_needed <= TPM_LIMIT:
            _token_window.append((now, tokens_needed))
            return

        # Espera até liberar espaço na janela
        sleep_time = WINDOW_SECONDS - (now - _token_window[0][0]) + 0.1
        time.sleep(max(0.1, sleep_time))


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run_gpt(
    model: str,
    prompt_template: str,
    extracted_text: str,
    edital_name: str,
) -> tuple[str, str]:
    """
    Retorna (output_text, prompt_hash).
    Você controla o prompt em prompts/prompt_gpt4.txt.
    """
    client = OpenAI(api_key="")  # usa OPENAI_API_KEY do ambiente

    filled = (
        prompt_template
        .replace("{{NOME_EDITAL}}", edital_name)
        .replace("{{TEXTO_EXTRAIDO}}", extracted_text)
    )
    p_hash = hash_text(filled)

    # --- trava para não estourar TPM ---
    tokens_needed = count_tokens(filled, model=model)
    wait_for_tpm(tokens_needed)

    resp = client.responses.create(
        model=model,
        input=filled
    )
    return resp.output_text, p_hash
