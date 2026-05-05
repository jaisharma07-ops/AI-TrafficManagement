# PRD: Forensic AI — Vision-Language Traffic Incident Analyst

> **One-line pitch:** A multimodal AI system that doesn't just *detect* traffic incidents — it *explains* them in natural language, assigns fault, and auto-generates legal/insurance reports from raw video footage.

---

## 1. Project Overview

### 1.1 The Problem
Traditional traffic surveillance and dashcam systems can detect that *something* happened (collision, sudden braking) but cannot reason about **why** it happened, **who** was at fault, or generate human-readable documentation. Insurance adjusters and police investigators currently spend hundreds of hours per case manually reviewing footage, and human bias creeps into fault determination.

### 1.2 The Solution
A two-stage AI pipeline:
1. **Fast detector (YOLOv8)** continuously scans video and triggers only on anomalies (crashes, near-misses, sudden velocity changes).
2. **Vision-Language Model (LLaVA / Moondream2)** then performs "forensic reasoning" on the triggered clip — describing causality, identifying violations, and producing a structured incident report.

A **chat interface** lets users query the video naturally ("What color was the car that ran the light?") via retrieval over VLM-generated frame descriptions.

### 1.3 Why This Is High-Level
This is **not** a pixel-classification project. It is a multimodal-transformer reasoning system — the same class of technology behind GPT-4V, Gemini, and Claude. We are applying frontier multimodal AI to a real-world legal/civic problem.

### 1.4 Target Users
- **Primary:** Police investigators, insurance claims adjusters
- **Secondary:** Municipal traffic departments, fleet operators, legal teams
- **Demo audience:** Academic evaluators (this is a 2nd-year college project)

---

## 2. Functional Requirements

### 2.1 Must-Have (MVP — required for full marks)

| ID | Feature | Description |
|----|---------|-------------|
| F1 | Video Ingestion | User uploads `.mp4` / `.avi` traffic footage via Streamlit UI |
| F2 | Anomaly Detection | YOLOv8 detects vehicles + flags crashes / sudden deceleration |
| F3 | Forensic Trigger | When anomaly detected, extract window: **5 seconds before → 2 seconds after** |
| F4 | VLM Reasoning | Sampled frames from window sent to VLM with forensic prompt |
| F5 | Incident Report | Structured JSON + Markdown summary: time, vehicles, cause, fault, violation |
| F6 | Live Log Panel | Real-time event stream on dashboard (timestamped events) |
| F7 | Chat-with-Video | Natural language Q&A grounded in VLM-generated frame descriptions |
| F8 | Legal Summary Generator | One-click generation of insurance-ready / police-report-ready text |

### 2.2 Nice-to-Have (Stretch Goals)
- Vector DB (ChromaDB) for semantic search across long videos
- Bounding-box overlay rendering on the playing video
- Export report as PDF
- Multi-camera correlation
- Speed estimation from pixel motion + camera calibration

### 2.3 Out of Scope
- Live RTSP camera streaming (assume uploaded files only)
- Real production deployment / authentication
- Mobile app
- Training a custom VLM (we use pre-trained, quantized models)

---

## 3. Tech Stack (Locked Decisions)

| Layer | Tool | Rationale |
|-------|------|-----------|
| **VLM (the "brain")** | `llava-hf/llava-1.5-7b-hf` *or* `vikhyatk/moondream2` | Open-weight, runnable in 4-bit on consumer GPU / Colab |
| **Quantization** | `bitsandbytes` 4-bit (NF4) | Makes 7B model fit in ~6GB VRAM |
| **Object Detection** | `ultralytics` YOLOv8n / YOLOv8s | Fast, well-documented, pretrained on COCO (cars, trucks, persons) |
| **Video Processing** | OpenCV (`opencv-python`) | Frame extraction, decoding, drawing overlays |
| **Model Loading** | HuggingFace `transformers` | Standard interface for VLM inference |
| **Text LLM (for chat)** | `meta-llama/Llama-3.2-3B-Instruct` (quantized) | Lightweight, summarizes VLM descriptions for Q&A |
| **Vector DB** | ChromaDB *(optional, stretch)* | For "chat with video" retrieval at scale |
| **Frontend** | Streamlit | Python-native dashboard, fast to prototype |
| **Runtime** | Python 3.10+, PyTorch 2.x, CUDA (or Colab T4) | Standard ML stack |

> **Hardware target:** Should run on Google Colab Free Tier (T4 GPU, 16GB VRAM) or a laptop with ≥6GB VRAM. CPU-only fallback should work but be slow.

---

## 4. System Architecture & Logic Flow

```
┌─────────────────┐
│  Video Upload   │
│   (Streamlit)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Frame Iterator (cv2)   │  ◄── samples at e.g. 5 fps
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐       NO
│  YOLOv8 Anomaly Check   │ ────────► continue scanning
│  (crash / decel logic)  │
└────────┬────────────────┘
         │ YES
         ▼
┌─────────────────────────┐
│  Window Extractor       │
│  (-5s to +2s of event)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  VLM Forensic Reasoner  │ ◄── LLaVA / Moondream2
│  (per-frame + summary)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Description Store      │ ◄── list / ChromaDB
│  (timestamp → text)     │
└────────┬────────────────┘
         │
         ├─────────────► Live Log Panel (Streamlit)
         ├─────────────► Incident Report (JSON + MD)
         └─────────────► Chat Engine (Llama 3 + retrieval)
```

### 4.1 Anomaly Detection Logic
A frame is flagged as anomalous if **any** of:
1. YOLOv8 detects two or more vehicle bounding boxes whose IoU exceeds **0.3** for ≥2 consecutive frames *(proxy for collision)*.
2. A tracked vehicle's velocity (centroid Δ-pixels per frame) drops by **>70%** within 0.5s *(proxy for sudden deceleration)*.
3. A tracked vehicle's bounding box suddenly disappears *(proxy for occlusion / off-road)*.

Use a simple centroid-tracker (`scipy.spatial.distance.cdist`) — no need for DeepSORT for the MVP.

### 4.2 VLM Prompting Strategy
Send **3–5 keyframes** from the incident window (not every frame — too slow). Use this prompt:

```
You are a forensic traffic analyst. The following frames show a sequence of events
captured by a traffic camera. Frames are in chronological order.

Analyze and respond in JSON with these exact keys:
{
  "scene_description": "<one paragraph, factual>",
  "vehicles_involved": ["<color> <type>", ...],
  "sequence_of_events": ["<event 1>", "<event 2>", ...],
  "probable_cause": "<one sentence>",
  "violations_observed": ["<violation 1>", ...],
  "at_fault": "<vehicle description or 'unclear'>",
  "confidence": "<low | medium | high>"
}

Be objective. If unclear, say so. Do not invent details.
```

### 4.3 Chat-with-Video Strategy
Two modes, depending on stretch ambition:

- **Mode A (MVP):** Generate a 1-sentence VLM description for every 2-second clip → store in a list of `(timestamp, description)` tuples → on user question, pass the entire list + question to Llama 3 → return answer with timestamp citation.
- **Mode B (Stretch):** Embed each description with `sentence-transformers/all-MiniLM-L6-v2` → store in ChromaDB → on question, retrieve top-k most relevant clips → pass to Llama 3 for synthesis.

---

## 5. UI/UX Specification (Streamlit Dashboard)

The dashboard must feel like a **police forensic console**. Layout (3-pane):

```
┌────────────────────────────────────────────────────────────────────┐
│  🚨 FORENSIC AI — Traffic Incident Analyst         [Upload Video]  │
├──────────────────────────────────┬─────────────────────────────────┤
│                                  │  📋 LIVE LOG                    │
│                                  │  ─────────────────────────      │
│      [VIDEO PLAYER]              │  00:02 │ Normal traffic flow    │
│      with bounding boxes         │  00:04 │ ⚠ Sudden decel detected│
│      drawn on cars               │  00:05 │ 🚨 COLLISION EVENT     │
│                                  │  00:05 │ Analyzing frames...    │
│                                  │  00:07 │ Report generated       │
│                                  │                                 │
│                                  │  📄 INCIDENT REPORT             │
│                                  │  ─────────────────────────      │
│                                  │  Vehicles: Red SUV, Cyclist     │
│                                  │  Cause: Failure to stop at...   │
│                                  │  Fault: Red SUV                 │
│                                  │  [Download Legal Report ▼]      │
├──────────────────────────────────┴─────────────────────────────────┤
│  💬 CHAT WITH VIDEO                                                │
│  ────────────────────────────────────────────────────────────────  │
│  You: Generate a legal summary for the insurance company.          │
│  AI:  On [date] at 0:05, a Red SUV failed to stop at a marked     │
│       intersection, striking a cyclist who had the right of way... │
│                                                                    │
│  [Type your question...                                    ] [Send]│
└────────────────────────────────────────────────────────────────────┘
```

### 5.1 Visual Style
- Dark theme (`st.set_page_config(layout="wide")` + custom CSS).
- Monospace font for the live log to evoke a console feel.
- Color codes: 🟢 normal, 🟡 warning, 🔴 critical.

---

## 6. Data Models

### 6.1 Incident Report Schema (JSON)
```json
{
  "incident_id": "uuid-v4",
  "video_filename": "string",
  "timestamp_seconds": 5.2,
  "duration_analyzed_seconds": 7,
  "scene_description": "string",
  "vehicles_involved": ["Red SUV", "Blue cyclist"],
  "sequence_of_events": ["...", "..."],
  "probable_cause": "string",
  "violations_observed": ["Failure to stop at intersection"],
  "at_fault": "Red SUV",
  "confidence": "high",
  "generated_at": "ISO-8601 datetime",
  "model_used": "llava-hf/llava-1.5-7b-hf"
}
```

### 6.2 Frame Description Entry (for chat)
```json
{ "timestamp": 4.0, "description": "A red SUV approaches the intersection at moderate speed. A cyclist is visible in the bike lane on the right." }
```

---

## 7. Project Structure

```
forensic-ai/
├── README.md
├── requirements.txt
├── PRD.md                          # this file
├── app.py                          # Streamlit entry point
├── config.py                       # paths, model names, thresholds
├── core/
│   ├── __init__.py
│   ├── video_processor.py          # OpenCV wrapper: load, sample, extract windows
│   ├── anomaly_detector.py         # YOLOv8 + tracker + trigger logic
│   ├── vlm_engine.py               # LLaVA/Moondream loader + forensic prompting
│   ├── chat_engine.py              # Llama 3 + retrieval over descriptions
│   └── report_generator.py         # JSON + Markdown + legal-summary builders
├── ui/
│   ├── __init__.py
│   ├── live_log.py                 # log panel component
│   ├── video_panel.py              # player + bbox overlay
│   └── chat_panel.py               # chat box component
├── prompts/
│   ├── forensic_analysis.txt
│   ├── legal_summary.txt
│   └── chat_system.txt
├── samples/
│   └── demo_traffic.mp4            # demo video for grading
└── tests/
    └── test_anomaly.py
```

---

## 8. Implementation Roadmap (Phased)

### Phase 0 — "Vibe Check" (Day 1, no code)
- Open HuggingFace Spaces, find a hosted LLaVA demo.
- Upload 3–5 still images of crashes. Verify outputs are coherent.
- **Gate:** if VLM output is garbage, switch model before writing code.

### Phase 1 — Foundation (Days 2–3)
- Set up repo, `requirements.txt`, config.
- Implement `video_processor.py`: load video, iterate frames at sample rate, extract a window around timestamp `t`.
- Smoke test on `samples/demo_traffic.mp4`.

### Phase 2 — Anomaly Detection (Days 4–5)
- Integrate YOLOv8 (`ultralytics`).
- Implement centroid tracker + velocity calculation.
- Implement trigger logic from §4.1.
- Print "ANOMALY at t=X.Xs" to console — no UI yet.

### Phase 3 — VLM Forensic Reasoning (Days 6–8)
- Load LLaVA-1.5-7B in 4-bit (or Moondream2 if VRAM is tight).
- Implement `vlm_engine.analyze_window(frames) → dict`.
- Wire forensic prompt from `prompts/forensic_analysis.txt`.
- Verify JSON output parses reliably (add retry on parse failure).

### Phase 4 — Streamlit UI (Days 9–10)
- Build 3-pane layout from §5.
- Wire video playback + live log streaming.
- Render incident report on right pane.

### Phase 5 — Chat-with-Video (Days 11–12)
- Generate per-2-sec descriptions for full uploaded video.
- Wire Llama 3 chat over the description list (Mode A).
- Add chat panel to UI.

### Phase 6 — Polish & Demo Prep (Day 13)
- Bounding-box overlays on video.
- "Download Legal Report" button → `.md` and `.json`.
- Record demo screen capture as backup.

### Phase 7 — Stretch (if time)
- ChromaDB integration (Mode B chat).
- Speed estimation.
- Multi-incident handling per video.

---

## 9. Performance & Constraints

| Constraint | Target |
|------------|--------|
| Hardware floor | Google Colab T4 (16GB VRAM) |
| VLM inference latency | < 10s per incident window (acceptable for demo) |
| YOLOv8 inference | Real-time on T4 (≥25 fps at 640px) |
| Max video length (MVP) | 2 minutes |
| Description generation | Batched, async-friendly |
| Memory | VLM in 4-bit → ~5–6GB; YOLOv8n → <1GB |

> **Critical:** do NOT load the VLM and Llama 3 simultaneously on Colab Free unless 4-bit quantized. Lazy-load: keep YOLOv8 always-resident, swap VLM ↔ Llama on demand.

---

## 10. Demo Script (the "Full Marks" run)

The grader sees this in order:

1. **Open dashboard.** Empty state. "Upload a traffic video."
2. **Upload `demo_traffic.mp4`** (a clip with one clear violation — e.g., a vehicle running a red light).
3. **Live log streams** as scanning progresses:
   `00:02 Normal flow → 00:04 Sudden deceleration → 00:05 🚨 Collision`
4. **Right pane populates** with the incident report (vehicles, cause, fault).
5. **Demonstrator types in chat:** *"Who was at fault?"* → AI cites timestamp.
6. **Demonstrator types:** *"Generate a legal summary for the insurance company."* → AI produces a paragraph in formal tone.
7. **Click "Download Legal Report"** → `.md` file appears in browser downloads.
8. **Closing line:** "This is multimodal reasoning — the same class of AI behind GPT-4V — applied to a real civic problem."

---

## 11. Success Criteria

| Metric | Pass Bar |
|--------|----------|
| Pipeline runs end-to-end on demo video | ✅ required |
| VLM produces coherent forensic JSON | ✅ required |
| Live log shows event timeline | ✅ required |
| Chat answers at least 3 distinct questions correctly | ✅ required |
| Legal summary is grammatically clean and factually grounded | ✅ required |
| Demo runs in under 3 minutes total | ✅ required |
| Stretch: chat retrieves correct timestamp | bonus |
| Stretch: bounding boxes render on video | bonus |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLaVA-7B too heavy for available GPU | Fall back to Moondream2 (1.8B) — same interface, smaller |
| VLM hallucinates events not in video | Forensic prompt explicitly warns "do not invent"; show `confidence` field |
| YOLOv8 misses subtle anomalies | Add manual "Force Analyze at timestamp X" button as escape hatch |
| Streamlit slow on long videos | Process in background thread; show progress bar |
| JSON parse fails | Retry once with stricter prompt; fall back to free-text report |

---

## 13. Notes for Claude Code

- **Always activate a virtual environment** before installing.
- Use `requirements.txt` with **pinned versions** (PyTorch + transformers + bitsandbytes have compatibility quirks).
- Write `core/` modules to be **independently testable** — each should run from CLI for debugging before being wired into Streamlit.
- For Streamlit, use `st.session_state` to persist the analysis across reruns. Streamlit re-executes the script on every interaction — model loading must be wrapped in `@st.cache_resource`.
- Prefer **Moondream2 for first integration** (faster iteration), then swap to LLaVA-1.5 for final quality.
- Do not block the Streamlit main thread on long VLM calls — use `st.status()` and incremental updates.
- Generate the demo video synthetically if no real footage is available: stitch dashcam clips from public datasets (BDD100K, KITTI).

---

**End of PRD.**