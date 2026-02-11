# 🧠 NER Transformer Project

> **Author:** Saran742001  
> **Last Updated:** 2026-02-11


## 📌 Overview
This project is a **Named Entity Recognition (NER)** tool powered by the **Hugging Face Transformers** library. It utilizes the `dslim/bert-base-NER` model to identify and classify entities in text such as **Persons (PER)**, **Organizations (ORG)**, **Locations (LOC)**, and **Miscellaneous (MISC)**.

The project is designed with a modular structure, separating the prediction logic, preprocessing, and utility functions, making it easy to extend or integrate into other applications.

## 🚀 Features
- **State-of-the-Art Model**: Uses `BERT-base-NER` for high-accuracy entity detection.
- **Interactive CLI**: Simple command-line interface with color-coded outputs for better visualization.
- **Confidence Filtering**: Only displays entities above a specific confidence threshold (default: 0.6).
- **JSON Export**: Option to save extracted entities to a JSON file (`ner_output.json`).
- **Modular Codebase**: Clean separation of converting, chunking, and prediction logic.

## 📂 Project Structure

```bash
ner_transformer/
├── data/                  # Directory for storing dataset files
├── src/                   # Source code modules
│   ├── predict.py         # Loads model and handles inference
│   ├── preprocessing.py   # Text cleaning utilities
│   ├── postprocess.py     # Entity merging and cleaning utilities
│   └── text_chunker.py    # Logic for splitting long texts
├── main.py                # Main entry point (CLI application)
├── requirements.txt       # Project dependencies
└── ner_output.json        # Output file for extracted entities
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd ner_transformer
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the main script to start the interactive CLI:

```bash
python main.py
```

### Example Interaction:

1. **Run the script**:
   ```text
   🔤 Named Entity Recognition (NER)
   Type a sentence and press Enter
   📝 Enter text: Apple Inc. is planning to open a new store in San Francisco.
   ```

2. **View Results**:
   The output will display detected entities with their labels and confidence scores:

   ```text
   🔍 Named Entities Found:

   Entity: Apple Inc. | Label: ORG | Confidence: 0.99
   Entity: San Francisco | Label: LOC | Confidence: 0.98

   📊 Confidence Summary:

   ORG: Avg confidence = 0.99
   LOC: Avg confidence = 0.98
   ```

3. **Save Output**:
   You will be asked if you want to save the results to a JSON file.
   ```text
   💾 Save entities as JSON? (y/n): y
   ✅ Saved as ner_output.json
   ```

## 🧩 Modules Description

### `src/predict.py`
- Loads the `dslim/bert-base-NER` model using Hugging Face's `pipeline`.
- Defines `predict_entities(text)` which runs the model and filters results based on confidence scores.

### `src/preprocessing.py`
- Contains `clean_text(text)` to remove extra whitespace and prepare text for the model.

### `src/text_chunker.py`
- Contains `chunk_text(text, max_words)` to split long documents into smaller segments to avoid model token limits.

### `src/postprocess.py`
- Contains `clean_entities(entities)` to help merge duplicate entities or refine results (useful for custom pipelines).

## ⚙️ Configuration
You can modify `src/predict.py` to change the model or adjust the confidence threshold:

```python
# Change the model
MODEL_NAME = "dslim/bert-base-NER"

# Adjust threshold in function signature
def predict_entities(text, confidence_threshold=0.6):
    ...
```

## 📜 License
This project is open-source and available for educational and research purposes.
