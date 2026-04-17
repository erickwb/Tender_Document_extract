from pathlib import Path
import pandas as pd

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")
