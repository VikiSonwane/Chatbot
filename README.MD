# Chatbot Project

A simple AI-powered chatbot built with **Streamlit**, **LangChain**, and **Google Generative AI (Gemini)**.  
It supports **file uploads (PDFs, text files)**, document chunking, embeddings with FAISS, and text-to-speech output.

---

## 🚀 Features
- Chat with AI using **Google Gemini / DeepSeek** models.
- Upload PDF or text files for context-based Q&A.
- Document processing with **LangChain** (chunking + embeddings).
- **FAISS** vector database for efficient similarity search.
- **Text-to-Speech (TTS)** response playback.

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/VikiSonwane/Chatbot.git
   cd Chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## ▶️ Usage

Run the chatbot with Streamlit:
```bash
streamlit run chatbot.py
```

Upload a document, ask questions, and listen to spoken answers.

---

## 📂 Project Structure
```
Chatbot/
│── chatbot.py          # Main chatbot application
│── requirements.txt    # Dependencies
│── README.md           # Documentation
│── .env                # API keys (not pushed to GitHub)
```

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License.
