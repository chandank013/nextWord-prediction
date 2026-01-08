# 🧠 Next Word Prediction System

A deep learning–based **Next Word Prediction System** built using **LSTM / BiLSTM**, **TensorFlow**, and **Streamlit**.  
The application predicts the next word(s) given a sequence of text, generates full sentences, and provides an interactive web interface with Light/Dark mode.

---

## 🚀 Features

- 🔮 **Next Word Prediction**
- 🔢 **Top-3 predicted words with confidence scores**
- 📝 **Sentence generation**
- ⌨️ **Typing animation effect**
- 🌗 **Light / Dark theme toggle**
- 🎨 **Modern card-based UI**
- ⚡ **Fast inference using trained LSTM model**

---

## 🗂️ Project Structure

```
next-word-prediction/
├── data/ # Training dataset (text corpus)
├── models/ # Saved model & tokenizer
│ ├── next_word_prediction_model.h5
│ └── tokenizer.pickle
│
├── notebooks/ # Jupyter notebooks (training & experiments)
├── venv/ # Virtual environment (ignored in Git)
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files & folders
```
---

## 🧠 Model Details

- **Architecture**: LSTM / Bidirectional LSTM  
- **Framework**: TensorFlow (Keras API)  
- **Tokenizer**: Keras Tokenizer (saved using Pickle)  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Task**: Language modeling / next-word prediction  

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
bash

git clone https://github.com/chandank013/nextWord-prediction-system.git

cd nextWord-prediction-system

---

### 2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate

---

Linux / macOS

source venv/bin/activate

---

### 3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py

---

Then open your browser at:

http://localhost:8501

---