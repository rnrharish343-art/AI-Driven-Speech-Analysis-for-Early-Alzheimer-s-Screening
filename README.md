# AI-Driven Speech Analysis for Early Alzheimer's Screening

An end-to-end multi-modal deep learning framework and web application designed for non-invasive, early-stage Alzheimer's screening through the analysis of acoustic and linguistic speech patterns.

---

## 📌 Key Features

- **Dual-Modality Speech & Text Processing:** Combines acoustic signal representations with textual transcriptions to evaluate both vocal acoustics and semantic context.
- **Advanced Deep Learning Backbone:** Utilizes pre-trained **WavLM** for rich audio feature extraction and **Conformer** layers for long-range temporal dependencies.
- **Clinical Language Modeling:** Employs **ClinicalBERT** to parse semantic, syntactic, and structural speech markers indicative of cognitive decline.
- **Cross-Modal Attention Fusion:** Integrates acoustic and linguistic feature embeddings dynamically via cross-attention mechanisms for final classification.
- **FastAPI Web Interface:** Provides an interactive dashboard for rapid audio upload, real-time inference, and diagnostic scoring.

---

## 🏗️ Model Architecture
+-----------------------+
             |    Input Audio File   |
             +-----------+-----------+
                         |
         +---------------+---------------+
         |                               |
         v                               v
+-----------------+             +------------------+
|  Acoustic Path  |             |  Linguistic Path |
|   (WavLM +      |             |   (Whisper/STT + |
|   Conformer)    |             |   ClinicalBERT)  |
+--------+--------+             +--------+---------+
         |                               |
         +---------------+---------------+
                         |
                         v
         +-------------------------------+
         | Cross-Modal Attention Fusion  |
         +---------------+---------------+
                         |
                         v
         +-------------------------------+
         | Diagnostic Screening Output   |
         +-------------------------------+
## 🛠️ Tech Stack

- **Frameworks & Libraries:** PyTorch, Hugging Face Transformers, Torchaudio, Librosa
- **Backend API:** FastAPI, Uvicorn
- **Audio Processing:** WavLM, Conformer
- **NLP Backbone:** ClinicalBERT
- **Frontend / Interface:** HTML5, JavaScript, CSS3

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/ai-speech-alzheimers-detection.git](https://github.com/YOUR_USERNAME/ai-speech-alzheimers-detection.git)
