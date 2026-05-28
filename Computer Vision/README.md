<div align="center">

#  Knightro Face Recognition

### Software subsystem for the Knightro interactive animatronic mascot

*Recognizes enrolled UCF faculty to enable personalized greetings*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/status-Sprint%201-yellow.svg)](#-progress-tracker)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey.svg)](#)

</div>

---

## 📖 Overview

This is the software subsystem responsible for the **face recognition pipeline** of the Knightro animatronic — a senior design project at the University of Central Florida. When an enrolled faculty member walks up to Knightro, the system detects their face, looks up their identity from a small encrypted on-device database, and triggers a personalized greeting through the animatronic's speech and motion controllers.

The full pipeline runs **locally on a Raspberry Pi** with no cloud dependency. Privacy and consent are first-class design constraints, not afterthoughts.

---

## 📑 Table of Contents

- [Technology Pivot](#-technology-pivot-dlib--onnx-runtime)
- [Project Layout](#-project-layout)
- [Setup](#-setup-first-time)
- [Running the Tests](#-running-the-tests)
- [Baseline Benchmarks](#-baseline-benchmarks)
- [Progress Tracker](#-progress-tracker)
- [References](#-references)

---

## 🔄 Technology Pivot: dlib → ONNX Runtime

During Sprint 1, the original plan was to use the `face_recognition` library (which depends on `dlib`) for computing face embeddings. However, `dlib` 20.0.1 cannot compile on macOS 15 (Sequoia) + Apple Silicon due to two issues: missing C++ standard library headers (`<string>`, `<iosfwd>`) caused by a Python/compiler version mismatch, and a removed macOS header (`<fp.h>`) in dlib's bundled libpng. Multiple workarounds were attempted (Xcode CLT install, SDK path exports, PNG support disabled, Python upgrade from 3.10 to 3.14, manual source patching) — none fully resolved the build.

The team pivoted to **ONNX Runtime** with an **ArcFace ResNet100** model, which provides the same functionality with zero compilation required. ONNX Runtime ships pre-built wheels for macOS ARM, Linux ARM (Raspberry Pi), and Windows.

| | Original (dlib) | Current (ONNX Runtime) |
|---|---|---|
| **Library** | `face_recognition` (dlib wrapper) | `onnxruntime` + ArcFace ResNet100 |
| **Embedding dim** | 128 | 512 |
| **Install** | Compile from source (30-60 min, fails on macOS 15) | Pre-built wheel (`pip install`, < 1 min) |
| **Model size** | Bundled in dlib (~30 MB) | Separate download (~250 MB, one-time) |
| **Comparison metric** | Euclidean distance | Cosine similarity (matches ArcFace training objective) |
| **Pi 5 compatible** | Requires 30-60 min compile | Pre-built ARM64 wheel, instant install |
| **Accuracy (SD-12)** | Not tested (couldn't install) | 95% overall (100% TPR, 90% TNR) |

This pivot is documented as an engineering decision in the EDS. The change affected SD-9 (enrollment), SD-10 (storage format), and SD-11 (recognition comparison) but had no impact on SD-6 (privacy policy), SD-7 (detection), or SD-8 (tracking).

## 📁 Project Layout

```
Computer Vision/
├── src/                           # Python modules
│   └── face_detection.py          # SD-7: MediaPipe face detection
├── tests/                         # Manual and automated tests
│   └── test_face_detection.py     # SD-7: webcam demo + FPS baseline
├── models/                        # (auto-created) cached MediaPipe model files
├── data/                          # (auto-created) snapshots, encrypted embeddings
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

---

## 🚀 Setup (first time)

### 1. Create a virtual environment

A venv keeps this project's dependencies isolated from your system Python.

```bash
python -m venv .venv
```

Then activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You'll know it worked when your prompt shows `(.venv)` at the front.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs only the **core** packages (`opencv-python`, `mediapipe`, `numpy`) needed for SD-7 and SD-8. The recognition packages (`face_recognition`, `dlib`, `cryptography`) are commented out and will be enabled when we start SD-9.

> [!TIP]
> See [Known Issues](#-known-issues) below for the reason `dlib` is deferred and how to enable it later.

---

## 🧪 Running the Tests

### SD-7: Face Detection Webcam Demo

Opens your webcam, draws boxes around detected faces, and shows FPS and face count in the corner. This is the **laptop baseline benchmark** that Phase 2 will compare against Pi performance.

```bash
python tests/test_face_recognition.py
```

| Key | Action |
|-----|--------|
| `q` | Quit the demo |
| `s` | Save a snapshot of the current frame to `data/` |

> [!NOTE]
> On first run, this will download the MediaPipe face detector model (~230 KB) to `models/blaze_face_short_range.tflite`. After that it runs entirely offline.

> [!WARNING]
> **macOS users:** the first run will trigger a camera permission popup. Click **Allow**. If you're running from a terminal *inside* an editor (VS Code, etc.), the permission prompt is for **that app**, not Terminal — and you may need to fully quit and relaunch the editor after granting permission. See [Known Issues](#-known-issues) for details.

---

## 📊 Baseline Benchmarks

### SD-7 — Face Detection (laptop)

Recorded on the development laptop for the EDS comparison table. Phase 2 will re-run the same test on the Raspberry Pi and add a second column.

| Metric                       | 💻 Laptop (MacBook Air M1) | 🥧 Raspberry Pi (Phase 2) |
|------------------------------|----------------------------|---------------------------|
| Sustained FPS                | **30** (camera-capped)     | _TBD_                     |
| Confidence on frontal face   | **> 0.90**                 | _TBD_                     |
| Robust to ±45° head rotation | ✅ Yes                     | _TBD_                     |
| Hardware acceleration        | Apple M1 GPU               | _TBD_                     |

> [!NOTE]
> **Interpretation:** The laptop sustained the full 30 FPS that the webcam delivers, meaning the detector is **not the bottleneck on the laptop** — it could likely go faster with a higher-frame-rate camera. This is the expected ceiling for laptop performance and gives us a clean reference point for the Pi comparison.

---

## ✅ Progress Tracker

| Ticket  | Description                              | Status                                              |
|---------|------------------------------------------|-----------------------------------------------------|
| SD-6    | Privacy & Consent Policy                 | ✅ **Done**                                         |
| SD-7    | Face Detection (laptop)                  | ✅ **Done** — verified on MacBook Air M1, 30 FPS    |
| SD-8    | Face Tracking                            | ✅ **Done**                                       |
| SD-9    | Faculty Enrollment Process               |  ✅ **Done**     |
| SD-10   | Encrypted Embeddings Storage             | ✅ **Done**                                          |
| SD-11   | 1:N Recognition Comparison               | ✅ **Done**                                          |
| SD-12   | Accuracy Tests (laptop baseline)         |✅ **Done**                                           |
| SD-13   | Integration with greeting flow           | ⬜ Sprint 2 _(needs Pi hardware)_                    |
| SD-50   | Performance test plan                    | ✅ **Done** Sprint 1 (plan) / ⬜ Sprint 2 (execute)              |
| SD-54   | Safety validation plan                   | ✅ **Done** Sprint 1 (plan) / ⬜ Sprint 2 (execute)              |

## 📚 References

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR 2015.*
- Google AI Edge. *MediaPipe Face Detector documentation.* https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

- Hernandez, A. *Knightro Tech Memo 1: Raspberry Pi 4 Computer Vision Benchmarks.*

---

<div align="center">

**🛡️ Charge On!** ⚔️

</div>
