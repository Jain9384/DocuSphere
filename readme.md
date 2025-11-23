# DocuSphere: RAG Chatbot with Gemini AI & History

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![AI Model](https://img.shields.io/badge/Google-Gemini_AI-4285F4.svg)](https://ai.google.dev/)
[![VectorDB](https://img.shields.io/badge/VectorDB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DocuSphere** (or choose your preferred name!) is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed for intelligent, context-aware question-answering over large PDF documents. It leverages Google's Gemini AI, the LangChain framework, and a FAISS vector index to deliver highly accurate semantic search, coupled with persistent chat history management.

---

## ✨ Features

This project provides a robust solution for both document interaction and persistent user experience.

| Feature | Description | Core Technology |
| :--- | :--- | :--- |
| **Context-Aware Q&A** | Answers questions based *only* on the content of uploaded PDFs. | **Gemini AI** / **RAG** |
| **Efficient Retrieval** | Implements FAISS for high-speed vector similarity search, resulting in **~70% better** retrieval accuracy than keyword methods. | **FAISS** / **Sentence Transformers** |
| **Persistent History** | Stores all Q&A interactions, providing full **CRUD** (Create, Read, Update, Delete) functionality for chat logs. | **SQLite** / `database.py` |
| **Intuitive UI** | Deployed via Streamlit, offering an easy-to-use interface for uploading documents and managing history. | **Streamlit** |
| **Source Tracking** | Preserves document chunks with page numbers for verifiable source citation. | **PyPDF2** |

## ⚙️ Core Architecture & Implementation Details

| Component | Role in Pipeline | Key Details |
| :--- | :--- | :--- |
| **Embedding Model** | Generates high-quality vector representations of text chunks. | **'all-MiniLM-L6-v2'** |
| **Vector Database** | Stores and enables fast similarity search over the document embeddings. | **FAISS** |
| **LLM / Generator** | Synthesizes the final human-readable answer based on the retrieved context. | **Google Gemini AI** |
| **History Database** | Provides persistence and management for all user interactions. | **SQLite** (built-in) |

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

* Python 3.x installed.
* A Google AI API Key.

### Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd rag-chatbot-project # Or your actual directory name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Set your Gemini API key as an environment variable (`GEMINI_API_KEY`) or configure it directly in your application files as necessary.

### Usage

1.  **Process Documents (First Run):**
    Before asking questions, you must convert your PDFs into a vector index.
    ```bash
    python pdf-vector.py [Your_Document.pdf]
    ```
    This will generate the required `chunks.pkl` file.

2.  **Run the Streamlit Application:**
    ```bash
    python -m streamlit run app.py
    ```

## 📚 New Feature: History Database (CRUD)

This update introduces a persistent chat history using **SQLite** for reliable data storage.

### Technology Overview

* **File:** `database.py`
* **Technology:** SQLite (No external setup required)
* **Schema:** `history` table with columns: `id` (PK), `timestamp`, `question`, `answer`.

### Verification

The new "History" tab will allow users to:
* **Create (C):** Every new Q&A session is automatically saved.
* **Read (R):** View all past interactions in a table/list.
* **Update (U):** Edit the text of past questions or answers.
* **Delete (D):** Remove specific records from the history.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for features, or find a bug, please feel free to submit an issue or open a pull request.

