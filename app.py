import gradio as gr
import pandas as pd
from transformers import pipeline
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # CHANGED
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline

# --- 1. Load Data and Build the Knowledge Base ---
df = pd.read_csv("nrcri_faqs.csv")
loader = DataFrameLoader(df, page_content_column="answer_en")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# CHANGED: Specified model_name directly and used the new import
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

# --- 2. Initialize Models ---
# For local LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"temperature": 0.5, "max_length": 256},
)

qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Translation models
en_to_ig_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ig") # CHANGED: Specified "translation" task
ig_to_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ig-en") # CHANGED: Specified "translation" task

chat_history = []

def detect_language(text):
    # A simple heuristic for language detection
    # For a more robust solution, a dedicated language detection library can be used
    if any(igbo_char in text for igbo_char in "ịọụṅỊỌỤṄ"):
        return 'ig'
    return 'en'

# --- 3. The Core Chat Function ---
def farming_chat(message, history):
    global chat_history

    lang = detect_language(message)
    question_en = message
    if lang == 'ig':
        question_en = ig_to_en_translator(message)[0]['translation_text']

    result = qa_chain({"question": question_en, "chat_history": chat_history})
    answer_en = result['answer']

    if lang == 'ig':
        answer_ig = en_to_ig_translator(answer_en)[0]['translation_text']
        chat_history.append((question_en, answer_en))
        return answer_ig
    else:
        chat_history.append((question_en, answer_en))
        return answer_en

# --- 4. The Gradio Interface ---
iface = gr.ChatInterface(
    fn=farming_chat,
    title="Igbo Language Farming Advisor",
    description="Ask your farming questions in English or Igbo and get expert advice.",
    theme="soft",
    examples=[["What are the best yam varieties for my region?"], ["Kedu ụdị akpu kacha mma maka mpaghara m?"]]
)

if __name__ == "__main__":
    iface.launch(share=True) # Set share=True to get a public link
