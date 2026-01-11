# 📊 Chat With Your Dataset

A powerful, interactive Streamlit application that allows you to upload **CSV datasets** and "chat" with them using advanced Large Language Models (LLMs).

![App Screenshot](placeholder_for_screenshot.png)
*(Replace `placeholder_for_screenshot.png` with your actual application screenshot)*

## 🚀 Overview

This tool converts your raw data into insights by generating a statistical summary of your dataset and using **Retrieval-Augmented Generation (RAG)** to answer your questions. It supports multiple LLM providers, making it flexible for both personal and professional use.

## 🏗️ Architecture

The application follows a modern RAG pipeline architecture:

```mermaid
graph TD
    User[👤 User] -->|Uploads CSV| App[💻 Streamlit App]
    
    subgraph "Data Processing Phase"
        App -->|Reads File| Pandas[🐼 Pandas]
        Pandas -->|Generates Stats| Summary[📄 Data Summary]
        Summary -->|Converts to Doc| LCDoc[📝 LangChain Document]
    end
    
    subgraph "RAG Pipeline"
        LCDoc -->|Embeds Content| Embed[🧬 Embeddings Model]
        Embed -->|Stores Vectors| FAISS[🔍 FAISS Vector Store]
        
        User -->|Asks Question| Chain[⛓️ Retrieval Chain]
        Chain -->|Query| FAISS
        FAISS -->|Retrieved Context| LLM[🤖 LLM (Gemini/OpenAI/HF)]
        LLM -->|Generates Answer| App
    end
```

## ✨ Key Features

-   **Multi-Provider Support**:
    -   🟢 **Google Gemini**: Uses `gemini-1.5-flash` (Fast & Free Tier available).
    -   🔵 **OpenAI**: Supports GPT-3.5/4 (Standard API).
    -   🤗 **Hugging Face**: Supports **Free Tier** inference (e.g., `Zephyr`, `Mistral`) with local embeddings.
-   **Smart CSV Processing**: Automatically detects headers, row counts, missing values, and statistical distributions.
-   **Secure**: API keys are input via the sidebar and never stored.
-   **Chat-Optimized**: Automatically detects "Chat" or "Instruct" models (especially on Hugging Face) to format prompts correctly.

## 🛠️ Installation

1.  **Clone the Repository** (or download the files):
    ```bash
    git clone https://github.com/yourusername/chat-with-your-dataset.git
    cd chat-with-your-dataset
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This includes `langchain`, `streamlit`, `faiss-cpu`, and provider-specific SDKs.*

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## ⚙️ Configuration & Usage

1.  **Select Provider**: On the sidebar, choose between:
    *   **Google Gemini**: Requires a [Google AI Studio Key](https://aistudio.google.com/app/apikey).
    *   **OpenAI / Compatible**: Requires an OpenAI Key (or Base URL for local tools like LM Studio/Ollama).
    *   **Hugging Face**: Requires a [Free HF Token](https://huggingface.co/settings/tokens).
2.  **Upload Data**: Drag and drop your `.csv` file.
3.  **Analyze**: Review the automatic summary.
4.  **Chat**: Type standard questions like:
    *   *"What is the distribution of values in column X?"*
    *   *"Are there any missing values?"*
    *   *"Summarize the key trends in this data."*

## 📦 Dependencies

*   `streamlit`
*   `pandas`
*   `langchain` & `langchain-community`
*   `faiss-cpu`
*   `langchain-google-genai`
*   `langchain-openai`
*   `langchain-huggingface`
*   `sentence-transformers`
*   `tf-keras` (for compatibility)


