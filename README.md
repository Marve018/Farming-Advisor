

# English & Igbo Language Farming Advisor

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)![Framework: Gradio](https://img.shields.io/badge/Framework-Gradio-orange)![AI: Hugging Face](https://img.shields.io/badge/AI-Hugging%20Face-yellow)

An AI-powered, bilingual web application designed to bridge the agricultural knowledge gap for Igbo-speaking farmers in Nigeria.


## 📜 Table of Contents

- [English \& Igbo Language Farming Advisor](#english--igbo-language-farming-advisor)
  - [📜 Table of Contents](#-table-of-contents)
  - [🌾 The Problem](#-the-problem)
  - [💡 The Solution](#-the-solution)
  - [✨ Features](#-features)
  - [⚙️ Technologies Used](#️-technologies-used)
  - [🔧 System Architecture](#-system-architecture)
  - [🚀 Setup and Installation](#-setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Ensure Knowledge Base is Present](#4-ensure-knowledge-base-is-present)
    - [5. Run the Application](#5-run-the-application)
  - [📖 How to Use](#-how-to-use)
  - [📁 Project Structure](#-project-structure)
  - [🌱 Future Improvements](#-future-improvements)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🌾 The Problem

A vast amount of modern agricultural research, best practices, and technological advancements published by key institutes like the National Root Crops Research Institute (NRCRI) are in English. This creates a significant **language barrier** for many local, Igbo-speaking farmers in regions like Abia State. As a result, valuable knowledge that could improve crop yields, manage pests, and increase profitability remains inaccessible to the very people who need it most.

## 💡 The Solution

The **Igbo Language Farming Advisor** is a simple web application that directly addresses this challenge. It provides a conversational interface where farmers can ask agricultural questions in either English or Igbo. The application leverages AI models to:
1.  Understand the user's question, regardless of the language.
2.  Retrieve relevant, expert-backed information from a knowledge base built on NRCRI's research.
3.  Generate a conversational, easy-to-understand answer.
4.  Translate the answer into the user's preferred language.

This project aims to empower farmers by making crucial agricultural knowledge accessible, conversational, and available in their native language.

## ✨ Features

-   **Bilingual Support:** Fully functional in both English and Igbo.
-   **Conversational AI:** Uses a Large Language Model (LLM) to provide natural, helpful answers instead of just raw data.
-   **Expert Knowledge Base:** Pre-loaded with synthesized FAQs and best practices from the NRCRI on key crops like cassava, yam, and sweet potato.
-   **Simple Web Interface:** Built with Gradio for an intuitive, mobile-friendly chat experience.
-   **Automatic Language Detection:** Automatically detects the input language to provide a seamless user experience.

## ⚙️ Technologies Used

-   **Backend:** Python
-   **AI & ML Frameworks:**
    -   [LangChain](https://www.langchain.com/): For structuring the conversational retrieval pipeline (RAG).
    -   [Hugging Face Transformers](https://huggingface.co/docs/transformers/index): For accessing pre-trained models for translation and language understanding.
    -   [PyTorch](https://pytorch.org/): As the backend for running the AI models.
-   **Web UI:** [Gradio](https://www.gradio.app/)
-   **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss): For efficient similarity search to find relevant information.
-   **Embeddings:** [Sentence-Transformers](https://www.sbert.net/): For converting text into numerical vectors.

## 🔧 System Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture:

1.  **User Input:** A farmer asks a question in English or Igbo via the Gradio web interface.
2.  **Language Detection:** A simple function detects the input language.
3.  **Translation (if needed):** If the question is in Igbo, the `Helsinki-NLP` model translates it to English.
4.  **Information Retrieval:** The English question is converted into a vector embedding. This embedding is used to search the FAISS vector store for the most relevant documents from the `nrcri_faqs.csv` knowledge base.
5.  **Answer Generation:** The original question and the retrieved documents are passed to the `google/flan-t5-base` LLM. The model generates a conversational answer in English based on the provided context.
6.  **Translation (if needed):** If the original question was in Igbo, the generated English answer is translated back to Igbo.
7.  **Display Output:** The final answer is displayed to the user in the chat interface.

## 🚀 Setup and Installation

Follow these steps to run the project on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Farming-Advisor.git
cd Farming-Advisor
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

-   **Windows (Git Bash / MINGW64):**
    ```bash
    python -m venv farming_advisor_env
    source farming_advisor_env/Scripts/activate
    ```
-   **macOS / Linux:**
    ```bash
    python3 -m venv farming_advisor_env
    source farming_advisor_env/bin/activate
    ```

### 3. Install Dependencies

First, create a `requirements.txt` file in your project directory with the following content:

**`requirements.txt`**
```
gradio
transformers
torch
langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
pandas
sentencepiece
```

Now, install all the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 4. Ensure Knowledge Base is Present

Make sure the `nrcri_faqs.csv` file (containing the farming questions and answers) is in the root directory of the project.

### 5. Run the Application

```bash
python app.py
```
The first time you run the script, it will download the necessary AI models, which may take several minutes depending on your internet connection. Subsequent launches will be much faster.

Once running, open the local URL (e.g., `http://127.0.0.1:7860`) provided in the terminal in your web browser.

## 📖 How to Use

1.  Open the application link in your web browser.
2.  Type your farming-related question into the chat box at the bottom. You can ask in either English or Igbo.
3.  Press Enter or click the send button.
4.  The AI advisor will process your question and provide a detailed answer in the same language you used.

**Example Questions:**
-   *English:* "What is the best way to control pests in my yam farm?"
-   *Igbo:* "Kedu ụzọ kacha mma esi egbochi ahụhụ n'ubi ji m?"

## 📁 Project Structure

```
Farming-Advisor/
├── farming_advisor_env/   # Virtual environment directory
├── app.py                 # The main Python script for the Gradio application
├── nrcri_faqs.csv         # The knowledge base file with Q&A data
├── requirements.txt       # A list of Python dependencies for the project
└── README.md              # This README file
```

## 🌱 Future Improvements

-   [ ] **Speech-to-Text/Text-to-Speech:** Integrate voice input and output to assist farmers with low literacy levels.
-   [ ] **Expand Knowledge Base:** Continuously add more data from NRCRI and other agricultural bodies to cover more crops and topics.
-   [ ] **User Feedback:** Add a "thumbs up/down" feature to rate the quality of answers and collect data for fine-tuning the model.
-   [ ] **Deployment:** Deploy the application to a permanent cloud platform like Hugging Face Spaces for public access.
-   [ ] **Image Support:** Allow farmers to upload images of diseased plants for AI-based diagnosis.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or want to fix a bug, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🙏 Acknowledgments

-   This project is powered by the invaluable research and publications of the **National Root Crops Research Institute (NRCRI), Umudike**.
-   Gratitude to the teams behind **Hugging Face**, **LangChain**, and **Gradio** for their incredible open-source tools that make projects like this possible.
