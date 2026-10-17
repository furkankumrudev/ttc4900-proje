"""Ortam uyumluluk kontrolü (PUBLISH_RULES['compat']): app.load + app.predict_proba ile verilen
train metinlerinin kalibre olasılıklarını npz'e yazar. Metinler tek tek işlenir (padding farkı
yok). ttc4900 paketine bağımlı değildir; ZeroGPU sürümleriyle kurulmuş ayrı bir ortamda da
çalışır. Çıktıda metin yok, yalnız row_id.

    python space/compat_check.py --model-dir outputs/models/42 \
        --calibration outputs/publish/model/calibration.json --texts <json> --out <npz>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tok, model, T, labels = app.load(a.model_dir, calibration_path=a.calibration)
    rows = json.loads(Path(a.texts).read_text(encoding="utf-8"))
    probs = np.vstack([app.predict_proba([r["text"]], tok, model, T).numpy() for r in rows])
    np.savez(
        a.out,
        row_id=np.array([r["row_id"] for r in rows], dtype=np.int64),
        probs=probs.astype(np.float64),
        labels=np.array(labels),
        torch=np.array(torch.__version__),
        transformers=np.array(transformers.__version__),
    )
    print(a.out, probs.shape, torch.__version__, transformers.__version__)


if __name__ == "__main__":
    main()
