# ⚖️ Forensic AI: Vision-Language Traffic Incident Analyst

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRD Compliant](https://img.shields.io/badge/PRD-Compliant-success)](PRD.md)

A multimodal AI pipeline designed to ingest raw traffic footage, detect road anomalies via computer vision, reconstruct incidents using lightweight Vision-Language Models (VLMs), and automatically generate legally binding insurance reports and fault-assessment summaries.
---
## 🛠️ System Architecture

The pipeline processes video streams sequentially, offloading heavy VLM reasoning to an event-driven trigger system to maximize runtime efficiency:

```mermaid


graph TD
    A[Raw Video Upload] --> B[Frame Iterator cv2]
    B --> C[YOLOv8n Anomaly Engine]
    C -->|Collision / Decel / Disappearance| D[Trigger Event]
    D --> E[Extract Window: -5s to +2s]
    E --> F[Keyframe Sampler: 4 Frames]
    F --> G[VLM Core: Moondream2 / LLaVA]
    G --> H[Forensic JSON Generation]
    H --> I[Report Engine: MD + Legal Summary]
    I --> J[Streamlit Dashboard & Chat Retrieval]
    # Clone and navigate to project root
Quickstart (Local Environment)
Follow these steps to spin up the local Streamlit dashboard on a standard CPU machine.

1. Environment Setup
    
cd "d:\AI project"

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install PyTorch CPU wheels & dependencies

pip install --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
forensic-ai/
├── app.py                      # Main Streamlit Dashboard application entry point
├── config.py                   # System-wide hyperparameters, IO paths, and confidence thresholds
├── requirements.txt            # Pinned package dependencies
│
├── core/                       # Core functional pipeline modules
│   ├── video_processor.py      # CV2 frame streaming, slicing, and temporal indexing
│   ├── anomaly_detector.py     # YOLOv8 target extraction and custom centroid tracking
│   ├── vlm_engine.py           # VLM prompt structuring and response parsers
│   ├── chat_engine.py          # Conversational context memory and RAG over scenes
│   └── report_generator.py     # Legal prose formatting and deterministic JSON schema validation
│
├── ui/                         # Streamlit component views
│   ├── live_log.py             # Real-time console pipeline output streaming
│   ├── video_panel.py          # Frame viewer and bounding box overlays
│   └── chat_panel.py           # Conversational forensic assistant interface
│
├── prompts/                    # System prompts and JSON-schema constraints for LLM/VLM
├── samples/                    # Target directories for source files and demonstration footage
├── scripts/                    # Utilities for data preparation and synthetic generation
├── notebooks/                  # Notebook runtimes (Colab workspace)
└── tests/

# Mech-Verse

<div align="center">

  **Where Mechanical Engineering Meets Digital Innovation.**

  [![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat&logo=vercel)](https://mech-verse.vercel.app/)
  [![Tech Stack](https://img.shields.io/badge/Stack-Next.js%20%7C%20React%20%7C%20Tailwind-blue?style=flat)](https://mech-verse.vercel.app/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  *Mech-Verse is an immersive, high-fidelity web ecosystem built to showcase the convergence of high-performance physical systems, advanced mechanical engineering concepts, and modern full-stack web architecture.*

  [Explore the Platform](https://mech-verse.vercel.app/) • [Report Bug](https://github.com/your-username/mech-verse/issues) • [Request Feature](https://github.com/your-username/mech-verse/issues)
</div>
---
## 🌌 Overview

Mech-Verse redefines how complex technical portfolios and engineering assets are experienced on the web. Engineered with a premium, low-friction, dark-matte aesthetic and cinematic motion design, the platform serves as a modern digital twin hub for high-performance vehicle dynamics, computer-aided design (CAD) architectures, and structural analysis data visualizations.

### Key Focus Areas
*   **Engineering Asset Management:** High-fidelity presentation of finite element analysis (FEA) datasets, fluid dynamics, and CAD files.
*   **Immersive UX Architecture:** Low-latency client-side rendering optimized for complex mathematical representations and fluid technical layouts.
*   **Developer Experience (DX):** Fully typed, componentized architecture designed for scalability and continuous deployment.

---


## 🛠️ Architecture & Tech Stack

The platform is engineered using a robust, decoupled frontend architecture designed for optimal performance, type-safety, and fluid layout rendering.

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Core Framework** | `Next.js 14+` (App Router), `React 18`, `TypeScript` | Server-side rendering (SSR), static site generation (SSG), and compile-time type safety. |
| **Styling & Theme** | `Tailwind CSS`, `CSS Modules` | Atomic design tokens, hyper-modern dark UI framework, utility-first responsiveness. |
| **Motion Design** | `Framer Motion` | Fluid structural transitions, micro-interactions, and hardware-accelerated animations. |
| **Hosting & CI/CD**| `Vercel` | Edge network distribution, instant deployments, and automated pipelines. |

---


## 🚀 Key Features

*   **Premium Visual Framework:** Hyper-modern, high-contrast dark UI featuring subtle aerodynamic neon accents and a carbon-fiber-inspired asset matrix layout.
*   **Engineering Modules Container:** Custom-built components ready to ingest and visualize real-world physical dynamics and multi-body physics simulation data.
*   **Zero-Lag Fluidity:** Optimized route-handling and state distribution ensure instant page transitions and seamless interaction on both mobile and desktop views.
---
## Getting Started

Follow these steps to spin up a local development environment.

### Prerequisites

Ensure you have Node.js installed (v18.x or higher recommended) along with a package manager (`npm`, `pnpm`, or `yarn`).

```bash
# Verify Node.js version
node -v
