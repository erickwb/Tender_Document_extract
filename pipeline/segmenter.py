import re

def normalize_ws(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def segment_by_page(pages: list[str]) -> list[dict]:
    out = []
    for idx, t in enumerate(pages, start=1):
        t = normalize_ws(t)
        if not t:
            continue
        out.append({
            "text": t,
            "page_start": idx,
            "page_end": idx,
        })
    return out


#segemntacao por secao:
import re

# Cabeçalhos típicos: "1. ...", "1.1 ...", "2.3.4 ...", com ou sem espaços após o ponto
PADRAO_SECAO = re.compile(r'^\s*\d+(?:\s*\.\s*\d+)*\s*\.[^\n]*', re.MULTILINE)

def segment_by_section(pages: list[str]) -> list[dict]:
    """
    Segmenta o documento em seções com base em cabeçalhos numéricos.
    Entrada: lista de páginas (strings).
    Saída: lista de dicts:
      {
        "text": <texto da seção>,
        "page_start": <página inicial aproximada>,
        "page_end": <página final aproximada>
      }

    Observação: como as seções são detectadas no texto concatenado, o mapeamento page_start/page_end
    é feito por interseção de offsets (aproximação consistente).
    """
    # --- monta texto concatenado e mapa de offsets por página ---
    texto = ""
    paginas_meta = []
    pos = 0

    for i, p in enumerate(pages, start=1):
        p = normalize_ws(p)
        if texto:
            texto += "\n"
            pos += 1
        start = pos
        texto += p
        pos += len(p)
        end = pos
        paginas_meta.append({"page": i, "start": start, "end": end})

    if not texto.strip():
        return []

    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    matches = list(PADRAO_SECAO.finditer(texto))
    if not matches:
        # Sem seção detectada: tudo vira uma única seção
        return [{
            "text": texto.strip(),
            "page_start": 1,
            "page_end": len(pages)
        }]

    def pages_covered(sec_start: int, sec_end: int) -> tuple[int, int]:
        covered = []
        for pg in paginas_meta:
            # interseção de [sec_start, sec_end) com [pg["start"], pg["end"])
            if not (sec_end <= pg["start"] or sec_start >= pg["end"]):
                covered.append(pg["page"])
        if not covered:
            return (1, 1)
        return (min(covered), max(covered))

    out = []
    for i, m in enumerate(matches):
        sec_start = m.start()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        trecho = texto[sec_start:sec_end].strip()
        if not trecho:
            continue

        p_start, p_end = pages_covered(sec_start, sec_end)

        out.append({
            "text": trecho,
            "page_start": p_start,
            "page_end": p_end
        })

    return out

### segmentacao por frase
import re
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize

# Garanta o punkt (uma vez no projeto, não precisa repetir)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    
def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")  # NBSP
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def segment_by_sentence(pages: list[str]) -> list[dict]:
    """
    Segmenta o documento em frases.
    Entrada: lista de páginas (strings).
    Saída: lista de dicts:
      {
        "text": <frase>,
        "page_start": <página>,
        "page_end": <página>
      }
    Observação: segmentação por frase é feita por linha -> sent_tokenize(linha),
    preservando contexto de página.
    """
    out: list[dict] = []

    for pageno, page_text in enumerate(pages, start=1):
        page_text = norm(page_text)
        if not page_text:
            continue

        # quebra por linhas (mantém a lógica do seu código base)
        for line in page_text.splitlines():
            line_n = norm(line)
            if not line_n:
                continue

            # tokeniza frases dentro da linha
            for sent in sent_tokenize(line_n):
                sent_n = norm(sent)
                if not sent_n:
                    continue

                out.append({
                    "text": sent_n,
                    "page_start": pageno,
                    "page_end": pageno,
                })

    return out