# ⚽ Football Match Analysis System

An AI-powered Football Match Analysis System developed using **YOLOv8 Nano, Python, Deep Learning, and Gradio**. The system uses computer vision to analyze football match footage and generate useful analytical insights through an interactive interface.

## 📌 Project Overview

The Football Match Analysis System is designed to analyze football match footage using the **YOLOv8 Nano (YOLOv8n)** object detection model.

The project uses a lightweight YOLOv8 model for efficient football video analysis. Python-based analytics are integrated with a Gradio frontend to provide users with an interactive way to access the system.

Model development and experimentation were performed using Google Colab, while the frontend application was developed using Gradio.

## 🎯 Main Objectives

- Apply deep learning and computer vision techniques to football match analysis.
- Use YOLOv8 Nano for object detection in football footage.
- Process football match videos using Python.
- Generate useful analytical outputs.
- Provide an interactive and user-friendly interface.

## ✨ Key Features

- YOLOv8 Nano-based object detection.
- Football match video analysis.
- Python-based analytics.
- Integration of trained model weights.
- Interactive Gradio frontend.
- Evaluation results and analysis.
- User-friendly interface for football match analysis.

## 🧠 Model Information

**Model:** YOLOv8 Nano (YOLOv8n)

**Model Architecture:** You Only Look Once (YOLO)

**Framework:** Ultralytics YOLOv8

**Task:** Object Detection

**Model Weights:** `best.pt`

YOLOv8 Nano is a lightweight object detection model suitable for computer vision applications where computational efficiency is important.

## 🛠️ Technologies Used

- Python
- Google Colab
- YOLOv8 Nano
- Ultralytics
- PyTorch
- OpenCV
- Gradio
- NumPy
- Pandas
- Matplotlib

## 📂 Project Structure

```text
Football-Match-Analysis-System/
│
├── Football_Match_Analysis.ipynb
├── gradio_app.py
├── app.py
├── analytics.py
├── best.pt
├── data_fixed.yaml
├── evaluation_results.csv
│
├── evaluation/
├── output/
├── outputs/
├── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Football-Match-Analysis-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Football-Match-Analysis-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Gradio Application

```bash
python gradio_app.py
```

Open the local Gradio URL displayed in the terminal to access the application.

## 📓 Google Colab Notebook

The model development and experimentation notebook is included in this repository:

`Football_Match_Analysis.ipynb`

The notebook can be opened and executed using Google Colab.

## 📊 Evaluation Results

The repository includes evaluation-related resources:

- `evaluation_results.csv`
- `evaluation/`

These files contain project evaluation results and supporting materials.

## 🖥️ Application Screenshots

Screenshots of the Gradio interface and football match analysis outputs are available in the `screenshots/` directory.

## 👩‍💻 Developed By

**Hiba Shoaib**

## 🎓 Project Purpose

This project was developed as an academic deep learning and computer vision project to explore the application of YOLOv8 Nano in football match analysis.

---

⭐ If you find this project interesting, feel free to explore the repository!
