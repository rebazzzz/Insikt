"""
Insikt – Journalist AI Suite
GPU‑accelerated, 100% local, free. Swedish/English bilingual.
Features: Background summarization, real‑time chat with document RAG, entity extraction, timeline, sentiment, export.
"""

import os
import tempfile
import pickle
import re
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# GPU detection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parallel loading
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Export
from docx import Document as DocxDocument
from fpdf import FPDF

# Analysis tools
from transformers import pipeline
import yake

# -------------------------------------------------------------------
# Configuration & Language
# -------------------------------------------------------------------
APP_NAME = "Insikt"
OLLAMA_MODEL = "llama3.2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 500

LANGUAGES = {
    "sv": {
        "title": "Insikt – Journalist-AI",
        "welcome": "Välkommen till Insikt",
        "howto": "Ladda upp PDF-filer i sidofältet och chatta sedan med dem nedan. Assistenten kan svara på frågor, sammanfatta och analysera dina dokument. Alla svar är baserade på dina dokument när möjligt, med källhänvisning.",
        "upload": "Ladda upp dokument",
        "process_btn": "Bearbeta dokument",
        "processing": "Bearbetar... Vänligen vänta. Inga andra åtgärder är möjliga just nu.",
        "success_docs": "✅ {} dokument laddade ({} stycken).",
        "error_no_docs": "Ladda upp dokument först.",
        "knowledge_ready": "✅ Kunskapsbas redo.",
        "chunks_loaded": "📚 {} stycken inlästa från {} dokument.",
        "chat_title": "💬 Chatta med dina dokument",
        "chat_input": "Ställ en fråga...",
        "sources": "📚 Källor",
        "summarize_title": "📝 Sammanfattningsverktyg",
        "summarize_btn": "Generera sammanfattning",
        "focus": "Fokus / instruktioner",
        "target_pages": "Målsidor för sammanfattning",
        "density": "Ord per sida",
        "style": "Utmatningsstil",
        "refine_btn": "Använd förfinad metod (bättre för långa dokument)",
        "summarizing": "Sammanfattar...",
        "success_summary": "✅ Sammanfattning klar.",
        "analysis_title": "🔍 Analysverktyg",
        "ner_extract": "🏷️ Extrahera enheter",
        "timeline": "📅 Skapa tidslinje",
        "keywords": "🔑 Extrahera nyckelord",
        "sentiment": "📊 Analysera sentiment",
        "export_title": "📤 Exportera",
        "bias_check": "⚖️ Kontrollera partiskhet",
        "translate": "🌍 Översätt till",
        "session_title": "💾 Session",
        "save_session": "Spara session",
        "load_session": "Ladda session",
        "settings": "⚙️ Inställningar",
        "language": "Språk",
        "summary_in_progress": "En sammanfattning pågår.",
        "summary_will_continue": "Sammanfattningen fortsätter på originalspråket.",
        "error_ollama": "Kunde inte ansluta till Ollama. Kontrollera att Ollama körs (```ollama serve```) och att modellen '{}' är nedladdad.",
        "error_gpu_memory": "GPU-minnet är otillräckligt. Försök med färre eller kortare dokument, eller kör på CPU.",
        "error_pdf_corrupt": "En eller flera PDF-filer är skadade eller oläsbara.",
    },
    "en": {
        "title": "Insikt – Journalist AI",
        "welcome": "Welcome to Insikt",
        "howto": "Upload PDFs in the sidebar, then chat with them below. The assistant can answer questions, summarize, and analyze your documents. All responses are grounded in your documents when possible, with sources cited.",
        "upload": "Upload Documents",
        "process_btn": "Process Documents",
        "processing": "Processing... Please wait. No other actions are possible at this time.",
        "success_docs": "✅ {} docs loaded ({} chunks).",
        "error_no_docs": "Please upload documents first.",
        "knowledge_ready": "✅ Knowledge base ready.",
        "chunks_loaded": "📚 {} chunks loaded from {} documents.",
        "chat_title": "💬 Chat with Your Documents",
        "chat_input": "Ask a question...",
        "sources": "📚 Sources",
        "summarize_title": "📝 Summarization Tool",
        "summarize_btn": "Generate Summary",
        "focus": "Focus / instructions",
        "target_pages": "Target pages for summary",
        "density": "Words per page",
        "style": "Output style",
        "refine_btn": "Use refine method (better for long docs)",
        "summarizing": "Summarizing...",
        "success_summary": "✅ Summary complete.",
        "analysis_title": "🔍 Analysis Tools",
        "ner_extract": "🏷️ Extract Entities",
        "timeline": "📅 Generate Timeline",
        "keywords": "🔑 Extract Keywords",
        "sentiment": "📊 Analyze Sentiment",
        "export_title": "📤 Export",
        "bias_check": "⚖️ Bias Check",
        "translate": "🌍 Translate to",
        "session_title": "💾 Session",
        "save_session": "Save Session",
        "load_session": "Load Session",
        "settings": "⚙️ Settings",
        "language": "Language",
        "summary_in_progress": "A summary is in progress.",
        "summary_will_continue": "The summary will continue in the original language.",
        "error_ollama": "Could not connect to Ollama. Please ensure Ollama is running (```ollama serve```) and the model '{}' is downloaded.",
        "error_gpu_memory": "GPU memory insufficient. Try with fewer or shorter documents, or run on CPU.",
        "error_pdf_corrupt": "One or more PDF files are corrupted or unreadable.",
    }
}

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
@st.cache_resource
def load_llm():
    try:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.3, num_predict=2048)
    except Exception as e:
        st.error(get_text("error_ollama").format(OLLAMA_MODEL))
        st.stop()

@st.cache_resource
def load_embeddings(model_name="all-MiniLM-L6-v2"):
    try:
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': DEVICE})
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error(get_text("error_gpu_memory"))
            st.stop()
        raise

@st.cache_resource
def load_ner_pipeline():
    try:
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=0 if DEVICE=="cuda" else -1)
    except Exception as e:
        st.warning("NER pipeline kunde inte laddas. Funktionen är inaktiverad." if st.session_state.lang=="sv" else "NER pipeline could not be loaded. Feature disabled.")
        return None

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if DEVICE=="cuda" else -1)
    except Exception as e:
        st.warning("Sentiment pipeline kunde inte laddas." if st.session_state.lang=="sv" else "Sentiment pipeline could not be loaded.")
        return None

def get_text(key):
    lang = st.session_state.get("lang", "sv")
    return LANGUAGES[lang].get(key, key)

def load_single_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = uploaded_file.name
        return pages
    except Exception as e:
        # Corrupt PDF
        st.error(get_text("error_pdf_corrupt") + f" ({uploaded_file.name})")
        return []
    finally:
        os.unlink(tmp_path)

def process_uploaded_files(uploaded_files, progress_bar, status_text):
    docs = []
    total_files = len(uploaded_files)
    status_text.text(f"{get_text('progress_reading')} (0/{total_files})")
    
    pages_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_single_pdf, f): f for f in uploaded_files}
        for i, future in enumerate(as_completed(futures)):
            try:
                pages = future.result()
                pages_list.extend(pages)
            except Exception as e:
                st.error(f"Error loading PDF: {e}")
            progress_bar.progress((i+1)/total_files)
            status_text.text(f"{get_text('progress_reading')} ({i+1}/{total_files})")
    
    if not pages_list:
        st.error("Inga sidor kunde laddas från PDF-filerna." if st.session_state.lang=="sv" else "No pages could be loaded from the PDFs.")
        return []
    
    status_text.text(get_text("progress_chunking"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages_list)
    return chunks

def build_vectorstore(chunks, embeddings, progress_bar, status_text):
    status_text.text(get_text("progress_indexing"))
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    try:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    except Exception as e:
        st.error(f"Failed to build vector store: {e}")
        return None
    progress_bar.progress(1.0)
    return vectorstore

# ---------- BACKGROUND SUMMARIZATION ----------
class SummaryThread(threading.Thread):
    def __init__(self, docs, llm, target_pages, focus, style, words_per_page, lang, progress_callback):
        super().__init__()
        self.docs = docs
        self.llm = llm
        self.target_pages = target_pages
        self.focus = focus
        self.style = style
        self.words_per_page = words_per_page
        self.lang = lang
        self.progress_callback = progress_callback
        self.result = None
        self.error = None

    def run(self):
        try:
            target_words = self.target_pages * self.words_per_page
            if self.lang == "sv":
                initial_template = "Du är en undersökande journalist. Sammanfatta följande text med fokus på: {focus}. Inkludera viktiga fakta, namn, datum och sidhänvisningar. Text: {text}"
                refine_template = """
Du har en befintlig sammanfattning: {existing_summary}
Läs nu följande nya text och införliva eventuell NY viktig information i sammanfattningen.
Fokus: {focus}. Stil: {style}.
Om den nya texten tillför detaljer, namn, datum eller händelser, lägg till dem. Håll sammanfattningen sammanhängande och inom cirka {target_words} ord totalt.

Ny text: {text}

Förbättrad sammanfattning:
"""
                final_prompt = f"""
Du är en journalist och redaktör. Utifrån följande pågående sammanfattning, producera en slutlig, polerad sammanfattning på ungefär {target_words} ord.
Stil: {{style}}. Fokus: {{focus}}.
Inkludera källhänvisningar [Källa: filnamn, sida X] där möjligt.

Pågående sammanfattning:
{{current_summary}}

Slutlig sammanfattning:
"""
            else:
                initial_template = "You are an investigative journalist. Summarize the following text, focusing on: {focus}. Include key facts, names, dates, and page references. Text: {text}"
                refine_template = """
You have an existing summary: {existing_summary}
Now read the following new text and incorporate any NEW important information into the summary.
Focus: {focus}. Style: {style}.
If the new text adds details, names, dates, or events, add them. Keep the summary coherent and within about {target_words} words total.

New text: {text}

Improved summary:
"""
                final_prompt = """
You are a journalist and editor. Based on the following running summary, produce a final, polished summary of approximately {target_words} words.
Style: {style}. Focus: {focus}.
Include citations [Source: filename, page X] where possible.

Running summary:
{current_summary}

Final summary:
"""

            initial_prompt = PromptTemplate(template=initial_template, input_variables=["text", "focus"])
            refine_prompt = PromptTemplate(template=refine_template, input_variables=["existing_summary", "text", "focus", "style", "target_words"])

            current_summary = ""
            total = len(self.docs)
            for i, doc in enumerate(self.docs):
                if self.progress_callback:
                    self.progress_callback(i, total)
                text = doc.page_content
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                text_with_ref = f"[Från {source}, sida {page}]: {text}" if self.lang == "sv" else f"[From {source}, page {page}]: {text}"

                try:
                    if i == 0:
                        prompt = initial_prompt.format(text=text_with_ref, focus=self.focus)
                        current_summary = self.llm.invoke(prompt).content
                    else:
                        prompt = refine_prompt.format(
                            existing_summary=current_summary,
                            text=text_with_ref,
                            focus=self.focus,
                            style=self.style,
                            target_words=target_words
                        )
                        current_summary = self.llm.invoke(prompt).content
                except Exception as e:
                    self.error = str(e)
                    return

            # Final polish
            final_prompt_text = final_prompt.format(current_summary=current_summary, target_words=target_words, style=self.style, focus=self.focus)
            try:
                final_summary = self.llm.invoke(final_prompt_text).content
            except Exception as e:
                self.error = str(e)
                return
            self.result = final_summary
        except Exception as e:
            self.error = str(e)

def start_summary(docs, target_pages, focus, style, words_per_page, lang):
    llm = load_llm()
    thread = SummaryThread(docs, llm, target_pages, focus, style, words_per_page, lang, update_summary_progress)
    st.session_state.summary_thread = thread
    st.session_state.summary_progress = 0
    st.session_state.summary_total = len(docs)
    st.session_state.summary_running = True
    st.session_state.summary_result = None
    st.session_state.summary_error = None
    thread.start()

def update_summary_progress(current, total):
    st.session_state.summary_progress = current + 1
    st.session_state.summary_total = total

def check_summary_status():
    if st.session_state.get("summary_running", False):
        thread = st.session_state.get("summary_thread")
        if thread and not thread.is_alive():
            if thread.error:
                st.session_state.summary_error = thread.error
            else:
                st.session_state.summary_result = thread.result
                st.session_state.last_summary = thread.result
            st.session_state.summary_running = False
            st.session_state.summary_thread = None
            st.rerun()

# ---------- Chat with RAG ----------
def retrieve_context(query, vectorstore, k=7):
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"Sökning misslyckades: {e}" if st.session_state.lang=="sv" else f"Search failed: {e}")
        return []

def create_chat_prompt(history, context_docs, query, lang):
    if lang == "sv":
        system_template = """
Du är en hjälpsam, ärlig AI-assistent. Du kan komma åt användarens uppladdade dokument för att svara på frågor.
När du använder information från dokumenten MÅSTE du ange källan (filnamn och sidnummer) i ditt svar.
Om användarens fråga inte är relaterad till dokumenten eller du inte hittar relevant information, kan du svara baserat på din allmänna kunskap, men ange tydligt att du inte använder dokumenten.
Var alltid sanningsenlig och hitta inte på information.

Kontext från dokument:
{context}

Konversationshistorik:
{history}

Användare: {query}
Assistent:"""
    else:
        system_template = """
You are a helpful, honest AI assistant. You can access the user's uploaded documents to answer questions.
When you use information from the documents, you MUST cite the source (filename and page number) in your response.
If the user's question is not related to the documents or you don't find relevant information, you can answer based on your general knowledge, but clearly indicate that you are not using the documents.
Always be truthful and do not make up information.

Context from documents:
{context}

Conversation history:
{history}

User: {query}
Assistant:"""

    context_str = ""
    if context_docs:
        for doc in context_docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_str += f"[Källa: {src}, sida {page}]: {doc.page_content}\n\n" if lang == "sv" else f"[Source: {src}, page {page}]: {doc.page_content}\n\n"
    else:
        context_str = "Inga dokument tillgängliga eller ingen relevant information hittades." if lang == "sv" else "No documents available or no relevant information found."

    history_str = ""
    for msg in history[-10:]:
        role = "Användare" if isinstance(msg, HumanMessage) else "Assistent"
        if lang == "en":
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    prompt = system_template.format(context=context_str, history=history_str, query=query)
    return prompt

def chat_with_docs(query, history, vectorstore, llm, lang):
    context_docs = retrieve_context(query, vectorstore, k=7) if vectorstore else []
    prompt = create_chat_prompt(history, context_docs, query, lang)
    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        response = f"Fel vid generering av svar: {e}" if lang == "sv" else f"Error generating answer: {e}"
    return response, context_docs

# ---------- Analysis tools ----------
def extract_entities(text, ner_pipeline):
    if not ner_pipeline:
        return {}
    results = ner_pipeline(text[:10000])
    entities = {"PERSON": set(), "ORG": set(), "LOC": set()}
    for ent in results:
        if ent['entity_group'] in entities:
            entities[ent['entity_group']].add(ent['word'])
    return entities

def extract_timeline(docs):
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    timeline = []
    for doc in docs:
        text = doc.page_content
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            date = match.group()
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].replace('\n', ' ')
            timeline.append({
                "date": date,
                "context": context,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?")
            })
    timeline.sort(key=lambda x: x["date"])
    return timeline

def bias_check(summary, llm, lang):
    if lang == "sv":
        prompt = f"""
Du är en kritisk redaktör. Granska följande sammanfattning för eventuell partiskhet, saknad motbevisning eller överdriven tilltro till en enda källa.
Lista eventuella problem du hittar och föreslå vad en balanserad sammanfattning skulle innehålla.

Sammanfattning:
{summary}

Kritik:
"""
    else:
        prompt = f"""
You are a critical editor. Review the following summary for potential bias, missing counter‑evidence, or over‑reliance on a single source.
List any issues you find, and suggest what a balanced summary would include.

Summary:
{summary}

Critique:
"""
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Bias check failed: {e}"

def translate_text(text, target_lang, llm, source_lang):
    prompt = f"Översätt följande text till {target_lang}:\n\n{text}" if source_lang == "sv" else f"Translate the following text to {target_lang}:\n\n{text}"
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Translation failed: {e}"

def extract_keywords(text, top_n=10):
    try:
        kw_extractor = yake.KeywordExtractor(lan="en", top=top_n)
        keywords = kw_extractor.extract_keywords(text)
        return [kw for kw, score in keywords]
    except Exception as e:
        return [f"Error: {e}"]

def analyze_sentiment(text, pipe):
    if not pipe:
        return {"label": "N/A", "score": 0.0}
    try:
        return pipe(text[:512])[0]
    except Exception as e:
        return {"label": "Error", "score": 0.0}

# ---------- Export ----------
def export_text(text): return text.encode()
def export_docx(text):
    doc = DocxDocument()
    doc.add_paragraph(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        with open(tmp.name, "rb") as f: data = f.read()
    os.unlink(tmp.name); return data
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f: data = f.read()
    os.unlink(tmp.name); return data
def export_markdown(text): return text.encode()

# -------------------------------------------------------------------
# UI Custom CSS
# -------------------------------------------------------------------
def set_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #ffffff; color: #1e293b; }
    .css-1d391kg { background: #f8fafc; }
    .sidebar .sidebar-content { background: #f8fafc; }
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    .processing-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(3px);
        z-index: 999;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .processing-box {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        max-width: 400px;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .user-message {
        background: #3b82f6;
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background: #f1f5f9;
        color: #1e293b;
        margin-right: auto;
    }
    .source-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-radius: 0 8px 8px 0;
    }
    .stButton>button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover:not(:disabled) {
        background: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .stButton>button:disabled {
        background: #94a3b8;
        cursor: not-allowed;
        opacity: 0.6;
    }
    .stProgress > div > div > div > div { background-color: #3b82f6; }
    .info-box {
        background: #e0f2fe;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d1fae5;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    h1, h2, h3 { font-weight: 600; }
    .stExpander { border: 1px solid #e2e8f0; border-radius: 12px; margin-bottom: 1rem; }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .stChatInput {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    set_custom_css()

    # Initialize session state
    defaults = {
        "docs": None,
        "vectorstore": None,
        "chat_history": [],
        "last_summary": "",
        "lang": "sv",
        "processing": False,
        "summary_running": False,
        "summary_thread": None,
        "summary_progress": 0,
        "summary_total": 0,
        "summary_result": None,
        "summary_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Check if summary thread finished
    check_summary_status()

    # Sidebar
    with st.sidebar:
        st.markdown(f"## {get_text('title')}")
        st.caption(f"Körs på **{DEVICE.upper()}**")
        st.divider()

        # Language selector (always enabled)
        lang = st.selectbox(
            get_text("language"),
            options=["sv", "en"],
            format_func=lambda x: "Svenska" if x == "sv" else "English",
            key="lang_selector",
        )
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            st.rerun()

        st.divider()

        # Document upload
        st.markdown(f"### {get_text('upload')}")
        uploaded_files = st.file_uploader(
            " ",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=st.session_state.processing,
            help="Välj en eller flera PDF-filer" if lang=="sv" else "Select one or more PDF files"
        )

        if uploaded_files and st.button(get_text("process_btn"), disabled=st.session_state.processing, use_container_width=True):
            st.session_state.processing = True
            try:
                progress = st.progress(0)
                status = st.empty()
                chunks = process_uploaded_files(uploaded_files, progress, status)
                if chunks:
                    st.session_state.docs = chunks
                    st.success(get_text("success_docs").format(len(uploaded_files), len(chunks)))

                    progress.progress(0)
                    embeddings = load_embeddings()
                    vs = build_vectorstore(chunks, embeddings, progress, status)
                    if vs:
                        st.session_state.vectorstore = vs
                        status.text(get_text("knowledge_ready"))
            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                st.session_state.processing = False
                st.rerun()

        if st.session_state.docs:
            unique_sources = len(set(d.metadata['source'] for d in st.session_state.docs))
            st.info(get_text("chunks_loaded").format(len(st.session_state.docs), unique_sources))

        st.divider()

        # Summarization settings
        with st.expander(get_text("summarize_title"), expanded=False):
            focus = st.text_area(
                get_text("focus"),
                placeholder="t.ex. korruption, specifik person..." if lang=="sv" else "e.g., corruption, specific person...",
                height=80,
                disabled=st.session_state.processing
            )
            target_pages = st.slider(get_text("target_pages"), 1, 50, 5, disabled=st.session_state.processing)
            words_per_page = st.number_input(get_text("density"), 100, 500, 300, disabled=st.session_state.processing)
            style = st.selectbox(get_text("style"), ["neutral", "investigative", "dramatic", "formal"], disabled=st.session_state.processing)
            use_refine = st.checkbox(get_text("refine_btn"), value=True, disabled=st.session_state.processing)

        st.divider()

        # Session management
        with st.expander(get_text("session_title"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button(get_text("save_session"), disabled=st.session_state.processing, use_container_width=True):
                    if st.session_state.vectorstore:
                        try:
                            os.makedirs("session_data", exist_ok=True)
                            st.session_state.vectorstore.save_local("session_data/vectorstore")
                            with open("session_data/session.pkl", "wb") as f:
                                pickle.dump({
                                    "docs": st.session_state.docs,
                                    "chat_history": st.session_state.chat_history,
                                    "last_summary": st.session_state.last_summary,
                                    "lang": st.session_state.lang
                                }, f)
                            st.success("✅ Session sparad" if lang=="sv" else "✅ Session saved")
                        except Exception as e:
                            st.error(f"Save failed: {e}")
                    else:
                        st.warning("Inget att spara" if lang=="sv" else "Nothing to save")
            with col2:
                if st.button(get_text("load_session"), disabled=st.session_state.processing, use_container_width=True):
                    try:
                        if os.path.exists("session_data/session.pkl"):
                            embeddings = load_embeddings()
                            vs = FAISS.load_local("session_data/vectorstore", embeddings, allow_dangerous_deserialization=True)
                            st.session_state.vectorstore = vs
                            with open("session_data/session.pkl", "rb") as f:
                                data = pickle.load(f)
                            st.session_state.docs = data["docs"]
                            st.session_state.chat_history = data["chat_history"]
                            st.session_state.last_summary = data["last_summary"]
                            st.session_state.lang = data["lang"]
                            st.success("✅ Session laddad" if lang=="sv" else "✅ Session loaded")
                            st.rerun()
                        else:
                            st.error("Ingen sparad session" if lang=="sv" else "No saved session")
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        st.caption("All behandling sker lokalt och privat" if lang=="sv" else "All processing is 100% local and private.")

    # Main area
    # Processing overlay
    if st.session_state.processing:
        st.markdown("""
        <div class="processing-overlay">
            <div class="processing-box">
                <h3>""" + get_text("processing") + """</h3>
                <div class="stProgress"><div style="width:100%; background:#3b82f6; height:4px;"></div></div>
                <p>Var god vänta...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"## {get_text('welcome')}")
    st.markdown(f'<div class="info-box">{get_text("howto")}</div>', unsafe_allow_html=True)

    # Chat section
    st.markdown(f"### {get_text('chat_title')}")

    # Chat container
    chat_container = st.container()
    with chat_container:
        # Display messages
        messages_container = st.container()
        with messages_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

        # Chat input at bottom
        if prompt := st.chat_input(get_text("chat_input"), disabled=st.session_state.processing):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.markdown(f'<div class="chat-message user-message">{prompt}</div>', unsafe_allow_html=True)

            # Generate response
            with st.spinner("Tänker..." if lang=="sv" else "Thinking..."):
                llm = load_llm()
                history_lc = []
                for msg in st.session_state.chat_history[:-1]:
                    if msg["role"] == "user":
                        history_lc.append(HumanMessage(content=msg["content"]))
                    else:
                        history_lc.append(AIMessage(content=msg["content"]))

                answer, sources = chat_with_docs(
                    prompt,
                    history_lc,
                    st.session_state.vectorstore,
                    llm,
                    st.session_state.lang
                )

            # Add assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown(f'<div class="chat-message assistant-message">{answer}</div>', unsafe_allow_html=True)

            # Show sources if any
            if sources:
                with st.expander(get_text("sources")):
                    for doc in sources[:5]:
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f'<div class="source-box"><b>{src}</b> (s.{page})<br>{doc.page_content[:200]}...</div>' if lang=="sv" else f'<div class="source-box"><b>{src}</b> (p.{page})<br>{doc.page_content[:200]}...</div>', unsafe_allow_html=True)

    # Summarization tool
    with st.expander(get_text("summarize_title"), expanded=False):
        st.markdown("Generera en omfattande sammanfattning av dina dokument. Inställningar finns i sidofältet." if lang=="sv" else "Generate a comprehensive summary of your documents. Settings are in the sidebar.")
        
        if st.session_state.summary_running:
            st.markdown(f'<div class="info-box">{get_text("summarizing")} {st.session_state.summary_progress}/{st.session_state.summary_total}</div>', unsafe_allow_html=True)
            st.progress(st.session_state.summary_progress / st.session_state.summary_total)
        elif st.session_state.summary_error:
            st.markdown(f'<div class="warning-box">Fel: {st.session_state.summary_error}</div>', unsafe_allow_html=True)
        elif st.session_state.summary_result:
            st.markdown("### " + ("Sammanfattning" if lang=="sv" else "Summary"))
            st.write(st.session_state.summary_result)
            st.session_state.last_summary = st.session_state.summary_result

        if st.button(get_text("summarize_btn"), disabled=st.session_state.processing or st.session_state.summary_running or not st.session_state.docs, key="summarize_btn_main"):
            if not st.session_state.docs:
                st.error(get_text("error_no_docs"))
            else:
                start_summary(st.session_state.docs, target_pages, focus, style, words_per_page, st.session_state.lang)
                st.rerun()

    # Analysis tools
    with st.expander(get_text("analysis_title"), expanded=False):
        if not st.session_state.docs:
            st.warning("Ladda upp dokument först." if lang=="sv" else "Please upload documents first.")
        else:
            ner_pipeline = load_ner_pipeline()
            sent_pipeline = load_sentiment_pipeline()
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(get_text("ner_extract"), disabled=st.session_state.processing):
                    with st.spinner("Extraherar..." if lang=="sv" else "Extracting..."):
                        all_text = " ".join([d.page_content for d in st.session_state.docs[:50]])
                        entities = extract_entities(all_text, ner_pipeline)
                        for k, v in entities.items():
                            st.markdown(f"**{k}**")
                            st.write(", ".join(list(v)[:20]) + ("…" if len(v)>20 else ""))
            with col2:
                if st.button(get_text("timeline"), disabled=st.session_state.processing):
                    with st.spinner("Skapar tidslinje..." if lang=="sv" else "Building timeline..."):
                        timeline = extract_timeline(st.session_state.docs[:200])
                        if timeline:
                            for item in timeline[:30]:
                                st.markdown(f"**{item['date']}** – {item['context'][:150]}...  \n*{item['source']} s.{item['page']}*" if lang=="sv" else f"**{item['date']}** – {item['context'][:150]}...  \n*{item['source']} p.{item['page']}*")
                        else:
                            st.info("Inga datum hittades." if lang=="sv" else "No dates found.")
            with col3:
                if st.button(get_text("keywords"), disabled=st.session_state.processing):
                    with st.spinner("Extraherar nyckelord..." if lang=="sv" else "Extracting keywords..."):
                        text = st.session_state.last_summary if st.session_state.last_summary else st.session_state.docs[0].page_content
                        keywords = extract_keywords(text)
                        st.write("Toppnyckelord:" if lang=="sv" else "Top keywords:", ", ".join(keywords))

            if st.session_state.last_summary:
                if st.button(get_text("sentiment"), disabled=st.session_state.processing):
                    with st.spinner("Analyserar sentiment..." if lang=="sv" else "Analyzing sentiment..."):
                        result = analyze_sentiment(st.session_state.last_summary, sent_pipeline)
                        st.write(f"Sentiment: **{result['label']}** (konfidens: {result['score']:.2f})" if lang=="sv" else f"Sentiment: **{result['label']}** (confidence: {result['score']:.2f})")

    # Export tools
    with st.expander(get_text("export_title"), expanded=False):
        if not st.session_state.last_summary:
            st.info("Generera en sammanfattning först." if lang=="sv" else "Generate a summary first.")
        else:
            text = st.session_state.last_summary
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("📄 TXT", disabled=st.session_state.processing):
                st.download_button("Ladda ner" if lang=="sv" else "Download", export_text(text), "sammanfattning.txt" if lang=="sv" else "summary.txt")
            if col2.button("📃 DOCX", disabled=st.session_state.processing):
                st.download_button("Ladda ner" if lang=="sv" else "Download", export_docx(text), "sammanfattning.docx" if lang=="sv" else "summary.docx")
            if col3.button("📑 PDF", disabled=st.session_state.processing):
                st.download_button("Ladda ner" if lang=="sv" else "Download", export_pdf(text), "sammanfattning.pdf" if lang=="sv" else "summary.pdf")
            if col4.button("📝 MD", disabled=st.session_state.processing):
                st.download_button("Ladda ner" if lang=="sv" else "Download", export_markdown(text), "sammanfattning.md" if lang=="sv" else "summary.md")

            if st.button(get_text("bias_check"), disabled=st.session_state.processing):
                with st.spinner("Kontrollerar partiskhet..." if lang=="sv" else "Checking for bias..."):
                    llm = load_llm()
                    critique = bias_check(text, llm, st.session_state.lang)
                    st.markdown("### Partiskhetsgranskning" if lang=="sv" else "### Bias Critique")
                    st.write(critique)

            target_lang = st.selectbox(get_text("translate"), ["Svenska", "English", "Spanish", "French", "German"] if lang=="sv" else ["Swedish", "English", "Spanish", "French", "German"])
            if st.button("Översätt" if lang=="sv" else "Translate", disabled=st.session_state.processing):
                with st.spinner("Översätter..." if lang=="sv" else "Translating..."):
                    llm = load_llm()
                    translated = translate_text(text, target_lang, llm, st.session_state.lang)
                    st.markdown(f"### Översättning ({target_lang})" if lang=="sv" else f"### Translation ({target_lang})")
                    st.write(translated)

    # Footer (cleaned)
    st.divider()
    st.caption("Insikt – 100% lokalt, privat och gratis." if lang=="sv" else "Insikt – 100% local, private, and free.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Ett kritiskt fel uppstod. Starta om appen." if st.session_state.get("lang","sv")=="sv" else "A critical error occurred. Please restart the app.")
        st.exception(e)