# Forensic AI — Vision-Language Traffic Incident Analyst

> Multimodal AI that explains traffic incidents in natural language, assigns fault, and auto-generates legal/insurance reports from raw video footage.

This is the implementation of the project specified in [PRD.md](PRD.md).

---

## Architecture

```
Video upload → Frame iterator (cv2)
            → YOLOv8n anomaly check (collision / sudden decel / disappearance)
            → On trigger: extract -5s..+2s window
            → Sample 4 keyframes
            → VLM (Moondream2) forensic JSON
            → Incident report (JSON + Markdown + legal summary)
            → Live log + chat retrieval
```

Two runtime variants ship in this repo:

| Path | Use when | Model | Where |
|---|---|---|---|
| **Local CPU** (default) | Demo on a laptop with no NVIDIA GPU | Moondream2 (1.8B, fp16) | `app.py` |
| **Colab GPU** | Higher-quality demo (~10s per VLM call) | LLaVA-1.5-7B (4-bit) | `notebooks/colab_full_quality.ipynb` |

---

## Quickstart (local CPU, Windows)

```bash
cd "d:\AI project"
python -m venv venv
venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt

# Get a demo video (downloads a public CC-licensed clip)
python scripts/fetch_demo_video.py

# Or synthesize a tiny one offline
python scripts/make_demo_video.py

# Smoke-test the modules
pytest tests/ -q

# Run the dashboard
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Upload `samples/demo_traffic.mp4` and watch:

1. The live log streams scan progress.
2. An anomaly is detected and the right pane fills with the incident report (vehicles, cause, fault, confidence).
3. The chat panel answers questions like "Who was at fault?" or "Generate a legal summary".
4. **Download Legal Report** writes a `.md` and `.json` file to your browser downloads.

> **First-run note.** On first VLM invocation, ~3.7 GB of weights download from Hugging Face into `~/.cache/huggingface/`. Pre-warm the model before a graded demo by running `python -c "from core.vlm_engine import VLMEngine; VLMEngine().load()"`.

---

## Colab variant (full quality)

Open `notebooks/colab_full_quality.ipynb` in Google Colab, set runtime to **T4 GPU**, run all cells. The notebook installs CUDA wheels, loads LLaVA-1.5-7B in 4-bit (`bitsandbytes`), accepts a video upload, runs the same pipeline, and prints an inline incident report. Use this when grading hardware allows.

---

## Project structure

```
forensic-ai/
├── app.py                      Streamlit entry point
├── config.py                   Thresholds, model names, paths
├── requirements.txt            Pinned deps
├── core/
│   ├── video_processor.py      Frame iteration, window extraction
│   ├── anomaly_detector.py     YOLOv8 + centroid tracker + triggers
│   ├── vlm_engine.py           Moondream2 forensic reasoning
│   ├── chat_engine.py          Retrieval over descriptions
│   └── report_generator.py     JSON + Markdown + legal summary
├── ui/
│   ├── live_log.py
│   ├── video_panel.py
│   └── chat_panel.py
├── prompts/                    Prompt templates
├── samples/                    Demo videos
├── scripts/                    Demo-video fetch / synth
├── notebooks/                  Colab variant
└── tests/                      Unit tests for video + anomaly logic
```

---

## Hardware notes

- **CPU floor** is fine for the MVP — expect 30–60s per incident on a recent laptop.
- The local path uses **no quantization** (`bitsandbytes` requires CUDA).
- The Colab path uses 4-bit NF4 quantization to fit a 7B model in ~5 GB VRAM.
- Pre-warming the model is recommended before a live demo.

## License & data

All scripts in this repo are MIT-licensed. The demo video downloaded by `scripts/fetch_demo_video.py` is sourced from a CC-licensed public clip (URL recorded in `samples/README.md`). No copyrighted footage ships in this repository.
