from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    # I/O
    pdf_dir: Path
    out_dir: Path

    # Segmentação
    segmentation: str = "page"              # page | section
    max_chars_per_segment: int = 6000

    # Classificador
    clf_model_type: str = "distilbert"      # distilbert | bertimbau
    clf_model_dir: str = ""                 # pasta do modelo/tokenizer
    clf_batch_size: int = 16
    interest_threshold: float = 0.5

    # LLM (seletor)
    llm_provider: str = "openai"            # openai | gaia | sonnet

    # OpenAI
    llm_model: str = "gpt-4.1"
    prompt_path: Path = Path("prompt/prompt.txt")


    load_in_4bit = False
    temperature = 0
    top_p = 1.0
    max_new_tokens = 256

    # Claude (Anthropic)
    claude_model: str = "claude-sonnet-4-5"

