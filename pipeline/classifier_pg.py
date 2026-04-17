from pathlib import Path
import torch
from torch.nn.functional import softmax

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
)

class HFTextClassifier:
    def __init__(self, model_type: str, model_dir: str, device: str | None = None):
        self.model_type = model_type.lower().strip()
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Pasta do modelo não encontrada: {self.model_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.model_type == "distilbert":
            self._load_distilbert()
        elif self.model_type == "bertimbau":
            self._load_bertimbau()
        elif self.model_type == "roberta":
            self._load_roberta()
        else:
            raise ValueError("clf_model_type deve ser: distilbert | bertimbau | roberta")

        self.model.eval()
        self.model.to(self.device)
        self.num_labels = self.model.config.num_labels

    def _load_checkpoint_state_dict(self, weights_path: Path) -> tuple[dict, dict]:
        ckpt = torch.load(str(weights_path), map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        fixed = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "")
            fixed[nk] = v
        return fixed, ckpt if isinstance(ckpt, dict) else {}

    def _load_distilbert(self):
        weights_path = self.model_dir / "modelo_producao_distilbert.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights_path}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_dir))
        state_dict, meta = self._load_checkpoint_state_dict(weights_path)

        num_labels = int(meta.get("num_labels", 2))
        base_model = meta.get("model_name", "distilbert-base-uncased")

        self.model = DistilBertForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        self.model.load_state_dict(state_dict, strict=True)

    def _load_bertimbau(self):
        weights_path = self.model_dir / "modelo_producao_bertimbau.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        state_dict, meta = self._load_checkpoint_state_dict(weights_path)

        base_model = meta.get("model_name", "neuralmind/bert-base-portuguese-cased")
        num_labels = int(meta.get("num_labels", 2))

        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        self.model.load_state_dict(state_dict, strict=True)

    def _load_roberta(self):
        weights_path = self.model_dir / "modelo_producao_roberta.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights_path}")

        # tokenizer roberta (deve estar salvo na pasta: vocab/merges + tokenizer_config etc.)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)

        state_dict, meta = self._load_checkpoint_state_dict(weights_path)

        num_labels = int(meta.get("num_labels", 2))
        base_model = meta.get("model_name", "roberta-base")  # ajuste se treinou com outra


        # roberta específico (também funcionaria com AutoModelForSequenceClassification)
        self.model = RobertaForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
         )
   


    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 16) -> list[float]:
        probs: list[float] = []
        max_len = getattr(self.model.config, "max_position_embeddings", 512)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**enc).logits
            p = softmax(logits, dim=-1)

            if self.num_labels == 2:
                probs.extend(p[:, 1].detach().cpu().tolist())
            else:
                probs.extend(p.max(dim=1).values.detach().cpu().tolist())

        return probs
