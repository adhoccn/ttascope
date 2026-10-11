# TTAScope

> **Batch AI-generated-text detection with drift diagnostics — works on text from unknown
> LLMs, no target labels and no model access.**

A deployable service for batch AI-generated-text (AIGT) detection, built on the method of the
companion paper *"面向未知大语言模型的中文 AI 生成文本检测：判别漂移的可预测性与无标签批校正"*.
Author identity is withheld in this repository for double-blind review.

Given a **batch** of texts from an unknown LLM, TTAScope returns a per-item AI probability,
a batch-level verdict, and a drift diagnostic — without any label from the target domain and
without access to the target model.

**Live demo:** <http://106.54.192.87:8501/tta/> — paste a batch of Chinese text (or upload a
file) to see per-item verdicts and the drift diagnostic. No data is retained server-side.

## What it reports

- **Per-item** $P(\mathrm{AI})$ and verdict, with a confidence band.
- **Batch diagnostic**: $\Xi_{\mathrm{AI}}$, $\Xi_{\mathrm{human}}$, $\Delta_T$, predicted F1,
  and an explicit recommendation (correct / do not correct / not enough items).
- **Batch-size gate**: below the usable batch size for a feature set the request is rejected
  rather than silently answered, because the class-conditional estimate is unreliable there.

## Layout

```
detector/       Library: features, TTA alignment, drift diagnostics, model wrapper
app/            FastAPI service + single-page web UI
deploy/         install.sh, nginx and systemd templates, upload.sh
tests/          Smoke, remote, long-document, report-export and browser checks
```

## Install and run

```bash
pip install -r requirements.txt

# local
uvicorn app.server:app --host 127.0.0.1 --port 8511
# then open http://localhost:8511/
```

Models are loaded from `models/*.joblib` at startup (one file per feature set); override the
directory with `TTASCOPE_MODELS`. **Model files are not included** — train them on your own
corpus:

```python
from detector import TTADetector
det = TTADetector(feat='15D').fit(train_texts, train_labels)
det.save('models/ttascope_15D.joblib')
```

Behind nginx, serve the app under a sub-path (see `deploy/nginx-ttascope.conf`); the service
itself listens on `127.0.0.1:8511` and is not exposed directly.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Detection page |
| `/health` | GET | Health check |
| `/api/models` | GET | Loaded feature sets with their minimum / recommended batch sizes |
| `/api/detect` | POST | `{texts, feat, mode}` → per-item scores, batch verdict, diagnostics |

Minimum batch sizes are defined in one place in `detector/model.py` and delivered to the UI
through `/api/models`, so the gate cannot drift between server and front end.

## Tests

```bash
python tests/test_smoke.py                       # offline, no server needed
python tests/test_remote.py http://localhost:8501/tta
python tests/test_long.py   http://localhost:8501/tta
node tests/test_report.js                        # report export logic
node tests/browser_check.js http://localhost:8501/tta/   # end-to-end (needs a browser)
```

The remote tests default to `http://localhost:8501/tta`; pass a URL argument to point them
elsewhere.

## Known limits

- Requires a batch, not a single text. The method assumes the batch class prior is not
  extremely imbalanced; when it is, the diagnostic says so instead of correcting silently.
- Correction is a strictly monotone recalibration of the score: it moves the decision
  threshold to the right place, it does not improve ranking quality. Under a fixed false
  positive budget the detection rate is unchanged.

## License

MIT — see `LICENSE`.
