"""Gradio Space: BERTurk (seed 42) + temperature scaling, TTC4900 haber sınıflandırma.

Model Hub'dan `space_config.json`'daki sabit commit'le yüklenir. Private model için Space
secret'ı HF_TOKEN kullanılır: yalnız bu model reposunu okuyabilen fine-grained bir token
(kullanıcı Hub arayüzünden girer). Token yazdırılmaz ve hiçbir yere kaydedilmez.
Kaynak: github deposundaki space/app.py (ttc4900.publish ile yüklenir).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 512  # eğitimle aynı: baştan kesme
TOP_K = 3
NOTICE = (
    "Model yalnızca `savasy/ttc4900` veri setindeki haber metinleriyle ince ayarlandı. Başka "
    "türden ya da başka kaynaklardan gelen metinlerde tahminler ve olasılıklar güvenilir "
    "olmayabilir. Ajans adları ve tırnak biçimi gibi konu dışı işaretler tahmini etkileyebilir. "
    "Metnin ilk 512 token'ından sonrası kullanılmaz."
)


def load(source: str, revision: str | None = None, token: str | None = None,
         calibration_path: str | None = None):  # fmt: skip
    """source: Hub repo id ya da yerel klasör. Döner: (tokenizer, model, T, etiketler)."""
    tok = AutoTokenizer.from_pretrained(source, revision=revision, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, revision=revision, token=token
    ).eval()
    if calibration_path is None:
        if Path(source).is_dir():
            calibration_path = str(Path(source) / "calibration.json")
        else:
            from huggingface_hub import hf_hub_download

            calibration_path = hf_hub_download(
                source, "calibration.json", revision=revision, token=token
            )
    T = float(json.loads(Path(calibration_path).read_text(encoding="utf-8"))["temperature"])
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tok, model, T, labels


def predict_proba(texts: list[str], tok, model, T: float) -> torch.Tensor:
    """Kalibre olasılıklar: softmax(logits / T)."""
    enc = tok(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**enc).logits.float()
    return torch.softmax(logits / T, dim=-1)


def classify(text: str, state) -> dict[str, float]:
    tok, model, T, labels = state
    if not text or not text.strip():
        return {}
    p = predict_proba([text], tok, model, T)[0]
    return {labels[i]: float(p[i]) for i in range(len(labels))}


def build_ui(state):
    import gradio as gr

    with gr.Blocks(title="BERTurk TTC4900 haber sınıflandırma") as demo:
        gr.Markdown("# BERTurk TTC4900: Türkçe haber sınıflandırma")
        gr.Markdown(NOTICE)
        gr.Markdown(
            f"Olasılıklar sıcaklık ölçeklemesiyle kalibre edildi (T = {state[2]:.3f}); bu "
            "kalibrasyon yalnız eğitim verisinin dağılımı için geçerlidir."
        )
        inp = gr.Textbox(label="Haber metni", lines=8)
        out = gr.Label(num_top_classes=TOP_K, label="Tahmin (kalibre olasılık)")
        gr.Button("Sınıflandır").click(lambda t: classify(t, state), inp, out)
    return demo


def parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="BERTurk TTC4900 demo")
    ap.add_argument(
        "--model-dir",
        help="yerel model klasörü (ör. outputs/models/42); verilmezse "
        "space_config.json'daki Hub reposu ve commit'i kullanılır",
    )
    ap.add_argument("--calibration", help="yerel calibration.json (--model-dir ile zorunlu)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.model_dir:  # yerel demo: ağ yok, token yok
        if not a.calibration:
            raise SystemExit("--model-dir ile --calibration da verilmeli "
                             "(ör. outputs/publish/model/calibration.json)")  # fmt: skip
        state = load(a.model_dir, calibration_path=a.calibration)
    else:  # Space: Hub'dan sabit commit; HF_TOKEN = salt okunur Space secret'ı
        cfg = json.loads(Path(__file__).with_name("space_config.json").read_text(encoding="utf-8"))
        state = load(cfg["model_repo"], cfg["model_revision"], token=os.environ.get("HF_TOKEN"))
    build_ui(state).launch(share=False)  # yalnız 127.0.0.1; public paylaşım bağlantısı yok


if __name__ == "__main__":
    main()
