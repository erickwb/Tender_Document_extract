from pathlib import Path
from tqdm import tqdm
from ultils import run_gaia_per_row_and_aggregate

from config import PipelineConfig
from pdf_reader import pdf_to_pages
from segmenter import (
    segment_by_page, segment_by_section, normalize_ws, segment_by_sentence
)
from classifier_pg import HFTextClassifier
from llm_gpt4 import load_prompt_template, run_gpt
from llm_claud_sonnet import run_claude

from io_utils import ensure_dir, write_csv

def get_edital_name(pdf_path: Path) -> str:
    return pdf_path.stem

def segment(pages: list[str], mode: str) -> list[dict]:
    if mode == "page":
        return segment_by_page(pages)
    if mode == "section":
         return segment_by_section(pages)
    if mode == "sentence":
        return segment_by_sentence(pages)
    raise ValueError("segmentation deve ser: page | section | sentence")

def truncate_text(s: str, max_chars: int) -> str:
    s = normalize_ws(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0] + " ..."

def main(cfg: PipelineConfig) -> None:
    ensure_dir(cfg.out_dir)

    pdf_paths = sorted(cfg.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"Nenhum PDF encontrado em: {cfg.pdf_dir}")

    # 1) segmentação → segments.csv
    segments_rows = []
    seg_id = 0

    for pdf_path in tqdm(pdf_paths, desc="Lendo e segmentando PDFs"):
        edital = get_edital_name(pdf_path)
        pages = pdf_to_pages(pdf_path)
        segs = segment(pages, cfg.segmentation)

        for s in segs:
            seg_id += 1
            segments_rows.append({
                "texto_segmentado": truncate_text(s["text"], cfg.max_chars_per_segment),
                "nome_edital": edital,
                "tipo_segmentacao": cfg.segmentation,
                "segment_id": seg_id,
                "page_start": s["page_start"],
                "page_end": s["page_end"],
            })

    segments_csv = cfg.out_dir / "segments.csv"
    write_csv(segments_csv, segments_rows)

    # 2) classificação (modelo treinado) → relevantes.csv

    if cfg.clf_model_type == "distilbert":
        cfg.clf_model_dir = "C:\\Users\\Pichau\\Desktop\\projetos\\Tender_Document_extract\models\\tokenizer_producao_distilbert"
    elif cfg.clf_model_type == "bertimbau":
        cfg.clf_model_dir = "C:\\Users\\Pichau\\Desktop\\projetos\\Tender_Document_extract\models\\tokenizer_producao_bertimbau"
    elif cfg.clf_model_type == "roberta":
        cfg.clf_model_dir = "C:\\Users\\Pichau\\Desktop\\projetos\\Tender_Document_extract\models\\tokenizer_producao_roberta"

    clf = HFTextClassifier(cfg.clf_model_type, cfg.clf_model_dir)

    texts = [r["texto_segmentado"] for r in segments_rows]
    probs = clf.predict_proba(texts, batch_size=cfg.clf_batch_size)

    relevant_rows = []
    for r, p in zip(segments_rows, probs):
        label = 1 if p >= cfg.interest_threshold else 0
        if label == 1:
            relevant_rows.append({
                "texto_extraido": r["texto_segmentado"],
                "nome_edital": r["nome_edital"],
                "score": float(p),
                "label_pred": label,
                "tipo_segmentacao": r["tipo_segmentacao"],
                "segment_id": r["segment_id"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
            })

    relevantes_csv = cfg.out_dir / "relevantes.csv"
    write_csv(relevantes_csv, relevant_rows)

    # 3) agrega por edital e envia para LLM (GPT-4) → llm_outputs.csv
    prompt_template = load_prompt_template(cfg.prompt_path)

    # agrupar por edital
    by_edital = {}
    for r in relevant_rows:
        by_edital.setdefault(r["nome_edital"], []).append(r)

    llm_rows = []
    for edital, rows in tqdm(by_edital.items(), desc="Chamando LLM por edital"):
        # concatena textos relevantes (ordena por page_start e score opcionalmente)
        rows_sorted = sorted(rows, key=lambda x: (x["page_start"], -x["score"]))
        extracted = "\n\n".join([f"- {x['texto_extraido']}" for x in rows_sorted])
        extracted = truncate_text(extracted, 50000)  # limite de segurança

        if cfg.llm_provider == "openai":
            print(  "Usando LLM gpt4"  )
            out_text, p_hash = run_gpt(
                model=cfg.llm_model,
                prompt_template=prompt_template,
                extracted_text=extracted,
                edital_name=edital,
            )

        elif cfg.llm_provider == "sonnet":
                print(  "Usando LLM Claude Sonnet"  )
                out_text, p_hash = run_claude(
                model=cfg.claude_model,
                prompt_template=prompt_template,
                extracted_text=extracted,
                edital_name=edital,
                    )
        else:
            raise ValueError("llm_provider deve ser: openai | gaia")

    
        llm_rows.append({
            "nome_edital": edital,
            "prompt_hash": p_hash,
            "llm_model": cfg.llm_provider,
            "llm_output": out_text,
        })

    llm_csv = cfg.out_dir / "llm_outputs.csv"
    write_csv(llm_csv, llm_rows)

if __name__ == "__main__":
    cfg = PipelineConfig(
        pdf_dir=Path("C:\\Users\\Pichau\\Desktop\\projetos\\experimentos licia\\pipeline_mestrado\\editais"),
        out_dir=Path("C:\\Users\\Pichau\\Desktop\\projetos\\experimentos licia\\pipeline_mestrado\\editais\\extracao"),
        #page + bertimbau , section + distilbert, roberta + sentence
        segmentation="sentence",              # page|section|sentence
        clf_model_type="roberta",          # distilbert|bertimbau|roberta
        interest_threshold=0.5,
        #"gpt-4.1" ou gpt-5.2"" sonnet
        llm_provider="openai", #
        prompt_path=Path("C:\\Users\Pichau\Desktop\projetos\experimentos licia\pipeline_mestrado\prompt\prompt.txt"),
        clf_batch_size=16,
    )



    main(cfg)
