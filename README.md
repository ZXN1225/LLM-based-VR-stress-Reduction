# VR Personalized Art Intervention Agent

A multimodal LLM-driven VR restorative environment generation framework for stress reduction and emotional intervention.

This project integrates:

- Retrieval-Augmented Generation (RAG)
- Environmental Psychology (SRT & ART)
- Multimodal LLM auditing
- SDXL panorama generation
- Real-ESRGAN upscaling
- Music generation
- FastAPI + Gradio interactive interface

The framework generates personalized restorative VR scenes and therapeutic music according to user emotional descriptions.

---

# Features

- Personalized restorative scene generation
- Clinical environment reasoning based on SRT/ART
- Iterative auditing and refinement pipeline
- Panorama-oriented SDXL generation
- Seam refinement for 360° VR environments
- Physical metric extraction and evaluation
- Music generation integration
- Interactive Gradio web interface

---

# Project Structure

```text
.
├── AgentAPP.py
├── RAG_Agent.py
├── Auditing_Agent.py
├── Production_Agent.py
├── MetricsToolBox.py
├── Extraction_Database.py
└── Mu_Extraction.py
└── Convert_LoRA_Dataset.py
├── requirements.txt
├── models/
├── PictureData/
├── PictureBase/
├── static/
└── README.md
└── .env
```

---

# Environment Requirements

- Python 3.10 – 3.12
- CUDA-enabled GPU recommended
- NVIDIA RTX GPU with at least 12GB VRAM recommended

Tested Environment:

- Windows 10 / Ubuntu 22.04
- CUDA 12.1
- PyTorch 2.5.1 + cu121

---

# Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

---

## 2. Create environment

Using Conda:

```bash
conda create -n vrtherapy python=3.12
conda activate vrtherapy
```

---

## 3. Install PyTorch

Install according to your CUDA version.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Required Models and Assets

The following resources must be prepared manually:

```text
models/
├── lora/
│   └── Custome.safetensors
├── realesrgan/
│   └── RealESRGAN_x4plus.pth
├── pca.pkl
├── real_pano_mu.npy
└── real_pano_inv_cov.npy
```

You can use Mu_Extraction.py to get the npy and pkl file. 
For the custome safetensors, you can train it use your own picture dataset with kohya.

You also need:

```text
PictureData/
PictureBase/
```

for reference image storage and ChromaDB vector database.

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

CHROMA_OPENAI_API_KEY=your_openai_embedding_key

HF_TOKEN=your_huggingface_token

SUNO_API_KEY=your_suno_key
SUNO_API_BASE=your_suno_api_url
```

---

# Running the Application

```bash
python AgentAPP.py
```

Server:

```text
http://localhost:8000/gui
```

---

# Pipeline Overview

## Stage 1 — Restorative Reasoning

The RAG agent analyzes user emotional input and generates:

- stress analysis
- therapeutic strategy
- restorative environmental targets

---

## Stage 2 — Retrieval-Augmented Planning

The system retrieves reference restorative environments from the ChromaDB vector database.

---

## Stage 3 — Panorama Generation

SDXL + ControlNet generate restorative panorama scenes.

Features include:

- LoRA panorama adaptation
- Seam refinement
- Equirectangular consistency optimization

---

## Stage 4 — Physical Metrics Evaluation

The system extracts:

- brightness
- contrast
- greenery ratio
- sky ratio
- fractal dimension
- DS score
- Mahalanobis-based FAED score
- ......
---

## Stage 5 — Iterative Auditing

A multimodal LLM auditor evaluates:

- therapeutic alignment
- restorative quality
- environmental coherence
- visual safety

Failed generations are iteratively refined.

---

## Stage 6 — Final VR Output

The final scenes are:

- upscaled using Real-ESRGAN
- exported for Unity VR integration
- paired with generated therapeutic music

---

# Notes

- CUDA GPU is strongly recommended.
- Panorama generation is memory intensive.
- First-time model loading may take several minutes.
- Some APIs (e.g., Suno) may require additional access permissions.

---

# Citation

If you use this project in academic research, please cite the corresponding paper.

---

# License

This project is intended for academic and research purposes only.

