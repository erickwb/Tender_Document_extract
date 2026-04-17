import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
from llm_gaia import GaiaLLM, GaiaConfig, run_gaia

def safe_parse_json_list(s: str) -> tuple[list, str | None]:
    """
    Tenta extrair e parsear um JSON (lista) do texto retornado pela LLM.
    Retorna (lista, erro_ou_None).
    """
    if not s or not s.strip():
        return [], "empty_output"

    # tenta parse direto
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj, None
        return [], "json_is_not_list"
    except Exception:
        pass

    # fallback: tenta extrair primeiro bloco JSON [...], se a LLM vazou texto
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj, None
            return [], "extracted_json_is_not_list"
        except Exception as e:
            return [], f"json_parse_error: {repr(e)}"

    return [], "no_json_block_found"


def run_gaia_per_row_and_aggregate(
    gaia_client,
    prompt_template: str,
    relevantes_csv: Path,
    out_dir: Path,
    llm_model_name: str = "gaia",
) -> tuple[Path, Path]:
    """
    1) Lê relevantes.csv
    2) Chama GAIA para cada linha (cada trecho)
    3) Salva:
       - llm_outputs_rows.csv (uma chamada por linha)
       - llm_outputs_by_edital.csv (JSON agregado por edital)
    Retorna (path_rows, path_by_edital)
    """
    df = pd.read_csv(relevantes_csv)

    required_cols = {"texto_extraido", "nome_edital"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"relevantes.csv sem colunas obrigatórias: {missing}")

    # Saída por linha
    rows_out = []

    # Agregação por edital (lista de objetos JSON)
    agg_items: dict[str, list] = defaultdict(list)

    for _, r in df.iterrows():
        edital = str(r["nome_edital"])
        trecho = str(r["texto_extraido"])

        # Reaproveita seu run_gaia: ele injeta {{NOME_EDITAL}} e {{TEXTO_EXTRAIDO}}
        out_text, p_hash = run_gaia(
            gaia=gaia_client,
            prompt_template=prompt_template,
            extracted_text=trecho,
            edital_name=edital,
        )

        items, err = safe_parse_json_list(out_text)

        # acumula (sem consolidar; apenas concatena)
        if items:
            agg_items[edital].extend(items)

        rows_out.append({
            "nome_edital": edital,
            "segment_id": int(r["segment_id"]) if "segment_id" in df.columns else "",
            "page_start": int(r["page_start"]) if "page_start" in df.columns else "",
            "page_end": int(r["page_end"]) if "page_end" in df.columns else "",
            "prompt_hash": p_hash,
            "llm_model": llm_model_name,
            "llm_output_raw": out_text,
            "json_parse_error": err or "",
            "json_items_count": len(items),
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSV por linha
    path_rows = out_dir / "llm_outputs_rows.csv"
    pd.DataFrame(rows_out).to_csv(path_rows, index=False, encoding="utf-8-sig")

    # 2) CSV agregado por edital (JSON final por edital)
    by_edital_out = []
    for edital, items in agg_items.items():
        by_edital_out.append({
            "nome_edital": edital,
            "llm_model": llm_model_name,
            "items_json": json.dumps(items, ensure_ascii=False),
            "items_count": len(items),
        })

    path_by_edital = out_dir / "llm_outputs_by_edital.csv"
    pd.DataFrame(by_edital_out).to_csv(path_by_edital, index=False, encoding="utf-8-sig")

    return path_rows, path_by_edital
