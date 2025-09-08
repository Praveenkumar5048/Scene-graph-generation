# CV Project

## Overview
This project performs object detection and feature extraction using a pretrained Faster R-CNN model and GloVe word embeddings. It is organized for clarity and modularity.

## Setup Instructions

1. **Clone the repository**

2. **Create and activate a Python virtual environment**

   - **On Windows (PowerShell):**
     ```powershell
     python -m venv venv
     venv/Scripts/Activate
     ```

   - **On Ubuntu/Linux (bash):**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download GloVe embeddings**
   - Download the GloVe 6B embeddings from [https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)
   - Extract the files and place the `glove.6B.300d.txt` file inside a folder named `glove.6B` in the project root:
     ```
     CV Project/
     └── glove.6B/
         └── glove.6B.300d.txt
     ```

5. **Run the project**
   ```powershell
   python main.py
   ```

## File Structure
```
CV Project/
├── main.py
├── detector.py
├── features.py
├── glove_utils.py
├── config.py
├── requirements.txt
├── .gitignore
├── glove.6B/
│   └── glove.6B.300d.txt
└── test_image.png
```
