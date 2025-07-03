# UIT-Chatbot-CourseInfo

## Project Overview

This repository houses a Retrieval-Augmented Generation (RAG) chatbot designed to provide information about courses at the University of Information Technology (UIT). It leverages a combination of Python, FAISS, LlamaIndex, and potentially the Gemini model for question answering based on course information.  The chatbot aims to provide quick and accurate responses to student inquiries regarding course details, prerequisites, and other related questions.

## Prerequisites & Dependencies

Before running this project, ensure you have the following installed:

*   **Python 3.11:**  You should use Python 3.11 or lower because PyTorch with CUDA only work with these.
*   **Streamlit:** Streamlit for creating the user interface.
*   **faiss:** FAISS (Facebook AI Similarity Search) for efficient vector similarity search.
*   **llama-index:** LlamaIndex framework for building RAG applications.
*   **pandas:** For data manipulation.
*   **Other dependencies:** All necessary Python packages are listed in the `requirements.txt` file.

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/eiyuumiru/UIT-Chatbot-CourseInfo.git
    cd UIT-Chatbot-CourseInfo
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set the Gemini API Key (if using Gemini):**

    ```bash
    export GEMINI_API_KEY="<your-key>"
    ```
    Replace `<your-key>` with your actual Gemini API key.  This step is crucial if the RAG pipeline utilizes Gemini.

## CLI Usage

You can interact with the RAG pipeline via the command line:

- **Build FAISS index** from CSV:
  ```bash
  python rag_pipeline.py build --csv qa_full.csv \
      --chunk_size 150 --chunk_overlap 20
  ```
  **Arguments:**
  - `--csv`: path to CSV file (default: `qa_full.csv`)
  - `--chunk_size`: chunk size in tokens (default: 150)
  - `--chunk_overlap`: overlap tokens between chunks (default: 20)
  - `--device`: embedding device (`cpu`, `cuda`, or `auto`)

- **Query the index**:
  ```bash
  python rag_pipeline.py query "IT003 học gì?" \
      --top_k 10
  ```
  **Arguments:**
  - `question` (positional): the question string
  - `--top_k`: number of similar chunks to retrieve (default: 10)
  - `--device`: embedding device (`cpu`, `cuda`, or `auto`)

## Live Demo

You can try my live demo using Streamlit in: https://uit-chatbot-courseinfo.streamlit.app/