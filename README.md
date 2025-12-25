# 🤖 Question Answering using DistilBERT on bAbI Tasks

This project demonstrates how to fine-tune a pretrained `DistilBERT` model for **extractive Question Answering (QA)** using the [bAbI dataset](https://research.fb.com/downloads/babi/). The solution is built using both TensorFlow and PyTorch to showcase interoperability across frameworks.

---

## 📚 Dataset

- Source: [bAbI QA Tasks by Facebook AI Research](https://research.fb.com/downloads/babi/)
- Format: Hugging Face `datasets` format, loaded from disk
- Structure:
  - `story.text`: List of statements (facts)
  - `story.answer`: List of ground truth answers
  - `story.id`, `story.supporting_ids`, `story.type`: Metadata

Each QA example is created by using two supporting facts and one corresponding question-answer pair.

---

## 🧹 Preprocessing

1. **Flatten** nested fields using `.flatten()` from `datasets`
2. **Map** fields into a format containing:
   - `question`: 3rd sentence
   - `sentences`: concatenated 1st and 2nd sentence (used as context)
   - `answer`: 3rd answer from original list
3. **Compute Start & End Indices**:
   - Using `str.find(answer)` to locate answer span in `sentences`

---

## 🔄 Tokenization & Alignment

- Uses `DistilBertTokenizerFast` to tokenize `sentences` + `question` pair
- Converts character-level answer spans to token-level start/end positions
- Handles edge cases where char positions can't be mapped (e.g., `None` tokens)

---

## 🏋️ Training with TensorFlow

- Model: `TFDistilBertForQuestionAnswering` (`return_dict=False`)
- Data: Converted to TensorFlow `tf.data.Dataset`
- Batch size: 8
- Loss: Averaged cross-entropy on start and end positions
- Optimizer: Adam (`learning_rate=3e-5`)
- Epochs: 3
- Loss curve plotted using Matplotlib

---

## 🧪 Inference Example (TensorFlow)

Example:
```python
Question: "What is south of the bedroom?"
Context: "The hallway is south of the garden. The garden is south of the bedroom."
Output: "garden"

## 🧠 Training with PyTorch (Hugging Face Trainer API)
Model: DistilBertForQuestionAnswering

Dataset: Reformatted to torch.utils.data.Dataset using .set_format('pt')

Training:

Trainer API

Batch size: 8

Epochs: 3

F1 score computation for start and end prediction

✅ Evaluation Metrics
f1_start: Macro F1 score for predicted vs true start positions

f1_end: Macro F1 score for predicted vs true end positions

## 🚀 Inference (PyTorch)
Example:

Question: "What is east of the hallway?"
Context: "The kitchen is east of the hallway. The garden is south of the bedroom."
Output: "kitchen"
The model extracts the answer span from the tokenized inputs using argmax over the predicted logits.

## 📦 Requirements
Python >= 3.8

transformers

datasets

tensorflow

torch

sklearn

matplotlib

Install with:

pip install transformers datasets tensorflow torch scikit-learn matplotlib

## 📁 Project Structure
bash
Copy
Edit
.
├── data/                        # Disk-saved Hugging Face dataset
├── tokenizer/                  # DistilBERT tokenizer directory
├── model/
│   ├── tensorflow/             # TensorFlow fine-tuned model
│   └── pytorch/                # PyTorch fine-tuned model
├── train_tensorflow.py         # Training script (TF)
├── train_pytorch.py            # Training script (PT)
├── README.md



##📜 License
This project is released for educational and research use. Please refer to original licensing terms of the bAbI dataset and Hugging Face Transformers.


---

