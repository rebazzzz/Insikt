"""
Insikt – Journalist AI Suite
GPU‑accelerated, 100% local, free. Swedish/English bilingual.
Features: Background summarization, real‑time chat with document RAG, entity extraction, timeline, sentiment, export.
Now with user‑selectable GPU/CPU processing.
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

# Parallel loading
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
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

# LLM Models - Optimized for performance and accuracy
# Quantized models are smaller, faster, and use less memory while maintaining good quality
LLM_MODELS = {
    "llama3.2": {
        "display_name": "Llama 3.2 (Standard)",
        "display_name_en": "Llama 3.2 (Standard)",
        "description": "Bästa kvalitet men kräver mer minne. Rekommenderas för kraftfulla datorer.",
        "description_en": "Best quality but requires more memory. Recommended for powerful computers.",
        "speed": "⚡⚡",
        "quality": "⭐⭐⭐⭐⭐",
        "memory": "High",
    },
    "llama3.2:3b": {
        "display_name": "Llama 3.2 3B (Snabb)",
        "display_name_en": "Llama 3.2 3B (Fast)",
        "description": "Snabbare och mindre minneskrävande. Bra balans mellan hastighet och kvalitet.",
        "description_en": "Faster and less memory intensive. Good balance between speed and quality.",
        "speed": "⚡⚡⚡⚡",
        "quality": "⭐⭐⭐⭐",
        "memory": "Medium",
    },
    "llama3.2:1b": {
        "display_name": "Llama 3.2 1B (Turbo)",
        "display_name_en": "Llama 3.2 1B (Turbo)",
        "description": "Snabbast och minst minneskrävande. För svagare datorer eller quick tests.",
        "description_en": "Fastest and most memory efficient. For weaker computers or quick tests.",
        "speed": "⚡⚡⚡⚡⚡",
        "quality": "⭐⭐⭐",
        "memory": "Low",
    },
    "mistral:7b": {
        "display_name": "Mistral 7B",
        "display_name_en": "Mistral 7B",
        "description": "Alternativ modell med bra prestanda. Bra för engelska dokument.",
        "description_en": "Alternative model with good performance. Good for English documents.",
        "speed": "⚡⚡⚡",
        "quality": "⭐⭐⭐⭐",
        "memory": "High",
    },
}
DEFAULT_LLM_MODEL = "llama3.2:3b"  # Default to fast quantized model

# Chunking strategies
CHUNKING_STRATEGIES = {
    "semantic": {
        "display_name": "Smart (Semantisk)",
        "display_name_en": "Smart (Semantic)",
        "description": "Delar dokument baserat på mening och sammanhang. Bäst för längre dokument.",
        "description_en": "Splits documents based on meaning and context. Best for longer documents.",
        "icon": "🧠",
    },
    "fixed": {
        "display_name": "Standard (Fast storlek)",
        "display_name_en": "Standard (Fixed size)",
        "description": "Delar dokument i jämna delar. Snabbare men mindre flexibelt.",
        "description_en": "Splits documents into equal parts. Faster but less flexible.",
        "icon": "📏",
    },
}
DEFAULT_CHUNKING = "semantic"

OLLAMA_MODEL = DEFAULT_LLM_MODEL
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 500

# Embedding models - bge provides better quality than MiniLM
# User-friendly explanations added for non-technical users
EMBEDDING_MODELS = {
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "display_name": "Snabb (bge-small)",  # Fast (bge-small)
        "display_name_en": "Fast (bge-small)",
        "description": "Snabbare, mindre noggrann. Bra för testning och svaga datorer.",  # Faster, less accurate. Good for testing and weak computers.
        "description_en": "Faster, less accurate. Good for testing and weaker computers.",
        "speed": "⚡⚡⚡⚡⚡",  # 5 lightning bolts for fastest
        "quality": "⭐⭐",    # 2 stars for quality
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "display_name": "Balanserad (bge-base)",  # Balanced (bge-base)
        "display_name_en": "Balanced (bge-base)",
        "description": "Bästa valet! Balans mellan hastighet och noggrannhet.",  # Best choice! Balance between speed and accuracy.
        "description_en": "Best choice! Balance between speed and accuracy.",
        "speed": "⚡⚡⚡⚡",   # 4 lightning bolts
        "quality": "⭐⭐⭐",  # 3 stars for quality
    }
}
DEFAULT_EMBEDDING_MODEL = "bge-base"

# Default device (will be overridden by user choice)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        "device": "Processor (GPU/CPU)",
        "device_auto": "Auto – använd GPU om tillgängligt, annars CPU",
        "device_cuda": "GPU – tvinga GPU (kräver NVIDIA GPU med CUDA)",
        "device_cpu": "CPU – tvinga CPU (långsammare, men fungerar alltid)",
        "device_current": "Aktuell processor: {}",
        "device_warning_cuda_unavailable": "Varning: Ingen GPU hittades. Kör på CPU.",
        "device_info": "GPU är mycket snabbare för AI-arbete, men kräver ett kompatibelt NVIDIA-kort med CUDA installerat. CPU fungerar alltid men är långsammare.",
        "ollama_gpu_note": "Ollama måste konfigureras separat för GPU. Se Ollamas dokumentation.",
        "summary_in_progress": "En sammanfattning pågår.",
        "summary_will_continue": "Sammanfattningen fortsätter på originalspråket.",
        "error_ollama": "Kunde inte ansluta till Ollama. Kontrollera att Ollama körs (```ollama serve```) och att modellen '{}' är nedladdad.",
        "error_gpu_memory": "GPU-minnet är otillräckligt. Försök med färre eller kortare dokument, eller kör på CPU.",
        "error_pdf_corrupt": "En eller flera PDF-filer är skadade eller oläsbara.",
        "embedding_model": "Inbäddningsmodell",
        "embedding_model_info": "BGE-modeller ger bättre semantisk förståelse än MiniLM",
        # Document processing progress messages
        "progress_reading": "Läser dokument",
        "progress_chunking": "Delar upp i segment",
        "progress_indexing": "Bygger kunskapsbas",
        "progress_embedding": "Skapar semantiska vektorer",
        "progress_complete": "Klart!",
        "stage_reading": "Läser dokument {} av {}",
        "stage_chunking": "Delar upp dokument i {} segment",
        "stage_embedding": "Skapar vektorer för {} segment",
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
        "device": "Processing device",
        "device_auto": "Auto – use GPU if available, otherwise CPU",
        "device_cuda": "GPU – force GPU (requires NVIDIA GPU with CUDA)",
        "device_cpu": "CPU – force CPU (slower, but always works)",
        "device_current": "Current device: {}",
        "device_warning_cuda_unavailable": "Warning: No GPU found. Running on CPU.",
        "device_info": "GPU is much faster for AI workloads, but requires a compatible NVIDIA card with CUDA installed. CPU works everywhere but is slower.",
        "ollama_gpu_note": "Ollama must be configured separately for GPU. See Ollama documentation.",
        "summary_in_progress": "A summary is in progress.",
        "summary_will_continue": "The summary will continue in the original language.",
        "error_ollama": "Could not connect to Ollama. Please ensure Ollama is running (```ollama serve```) and the model '{}' is downloaded.",
        "error_gpu_memory": "GPU memory insufficient. Try with fewer or shorter documents, or run on CPU.",
        "error_pdf_corrupt": "One or more PDF files are corrupted or unreadable.",
        "embedding_model": "Embedding Model",
        "embedding_model_info": "BGE models provide better semantic understanding than MiniLM",
        # Document processing progress messages
        "progress_reading": "Reading documents",
        "progress_chunking": "Chunking documents",
        "progress_indexing": "Building knowledge base",
        "progress_embedding": "Creating semantic vectors",
        "progress_complete": "Complete!",
        "stage_reading": "Reading document {} of {}",
        "stage_chunking": "Chunking into {} segments",
        "stage_embedding": "Creating vectors for {} segments",
    }
}

# -------------------------------------------------------------------
# Helper functions (now device‑aware)
# -------------------------------------------------------------------
def get_text(key):
    lang = st.session_state.get("lang", "sv")
    return LANGUAGES[lang].get(key, key)

def resolve_device(choice):
    """Return actual device string based on user choice."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif choice == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            # Fallback with warning
            st.warning(get_text("device_warning_cuda_unavailable"))
            return "cpu"
    else:  # cpu
        return "cpu"

# We'll cache resources with a dependency on device choice so they reload when device changes.
@st.cache_resource(show_spinner=False)
def load_llm(_device_choice, _model_key=None):  # device_choice is not used directly but forces cache invalidation
    # Get the model from session state or use default
    model_key = _model_key or st.session_state.get("llm_model", DEFAULT_LLM_MODEL)
    try:
        return ChatOllama(model=model_key, temperature=0.3, num_predict=2048)
    except Exception as e:
        st.error(get_text("error_ollama").format(model_key))
        st.stop()

@st.cache_resource(show_spinner=False)
def load_embeddings(_device_choice, _embedding_model_key=None):
    device = resolve_device(_device_choice)
    # Get the embedding model key from session state or use default
    model_key = _embedding_model_key or st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    model_info = EMBEDDING_MODELS.get(model_key, EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])
    # Extract the actual model name from the dictionary
    model_name = model_info["model_name"] if isinstance(model_info, dict) else model_info
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error(get_text("error_gpu_memory"))
            st.stop()
        raise

@st.cache_resource(show_spinner=False)
def load_ner_pipeline(_device_choice):
    device = resolve_device(_device_choice)
    device_id = 0 if device == "cuda" else -1
    try:
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device_id)
    except Exception as e:
        st.warning("NER pipeline could not be loaded. Feature disabled." if st.session_state.lang=="en" else "NER pipeline kunde inte laddas.")
        return None

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(_device_choice):
    device = resolve_device(_device_choice)
    device_id = 0 if device == "cuda" else -1
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device_id)
    except Exception as e:
        st.warning("Sentiment pipeline could not be loaded." if st.session_state.lang=="en" else "Sentiment pipeline kunde inte laddas.")
        return None

# -------------------------------------------------------------------
# Rest of the helper functions (unchanged except using device-aware loaders)
# -------------------------------------------------------------------
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
        st.error(get_text("error_pdf_corrupt") + f" ({uploaded_file.name})")
        return []
    finally:
        os.unlink(tmp_path)

def semantic_chunking(pages, embeddings, status_text, threshold=0.5):
    """
    Semantic chunking using embeddings - splits documents based on 
    semantic similarity rather than fixed character counts.
    """
    from langchain_core.documents import Document
    
    status_text.text("Skapar semantiska segment..." if st.session_state.get("lang","sv")=="sv" else "Creating semantic segments...")
    
    # First, split into sentences using basic text splitting
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Small chunks for sentence-level analysis
        chunk_overlap=50,
        separators=[". ", "! ", "? ", "\n"]
    )
    
    # Get all sentences from all pages
    all_sentences = []
    for page in pages:
        sentences = sentence_splitter.split_text(page.page_content)
        for sent in sentences:
            if sent.strip():
                all_sentences.append({
                    "text": sent,
                    "source": page.metadata.get("source", "Unknown"),
                    "page": page.metadata.get("page", "?")
                })
    
    if not all_sentences:
        # Fallback to standard chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(pages)
    
    # Generate embeddings for all sentences
    status_text.text("Beräknar semantisk likhet..." if st.session_state.get("lang","sv")=="sv" else "Calculating semantic similarity...")
    
    # Batch embed for efficiency
    texts_to_embed = [s["text"] for s in all_sentences]
    embeddings_matrix = embeddings.embed_documents(texts_to_embed)
    
    # Group sentences into chunks based on similarity
    chunks = []
    current_chunk_texts = []
    current_chunk_sources = set()
    current_chunk_pages = set()
    
    for i, sent_info in enumerate(all_sentences):
        current_chunk_texts.append(sent_info["text"])
        current_chunk_sources.add(sent_info["source"])
        current_chunk_pages.add(str(sent_info["page"]))
        
        # Check if we should start a new chunk
        if len(current_chunk_texts) > 1:
            # Compare with previous sentence
            prev_embed = embeddings_matrix[i-1]
            curr_embed = embeddings_matrix[i]
            
            # Calculate cosine similarity
            import numpy as np
            prev_norm = np.linalg.norm(prev_embed)
            curr_norm = np.linalg.norm(curr_embed)
            if prev_norm > 0 and curr_norm > 0:
                similarity = np.dot(prev_embed, curr_embed) / (prev_norm * curr_norm)
            else:
                similarity = 0
            
            # If similarity is below threshold, finalize current chunk
            if similarity < threshold:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_texts)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": ", ".join(current_chunk_sources),
                        "page": ", ".join(current_chunk_pages)
                    }
                ))
                current_chunk_texts = []
                current_chunk_sources = set()
                current_chunk_pages = set()
    
    # Don't forget the last chunk
    if current_chunk_texts:
        chunk_text = " ".join(current_chunk_texts)
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "source": ", ".join(current_chunk_sources),
                "page": ", ".join(current_chunk_pages)
            }
        ))
    
    # If semantic chunking produced too few or too many chunks, adjust
    # Merge small chunks or split very large ones
    final_chunks = []
    min_chunk_size = 200
    
    for chunk in chunks:
        if len(chunk.page_content) < min_chunk_size and final_chunks:
            # Merge small chunk with previous
            final_chunks[-1] = Document(
                page_content=final_chunks[-1].page_content + " " + chunk.page_content,
                metadata=final_chunks[-1].metadata
            )
        elif len(chunk.page_content) > CHUNK_SIZE * 2:
            # Split large chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            split_chunks = splitter.split_text(chunk.page_content)
            for sc in split_chunks:
                final_chunks.append(Document(page_content=sc, metadata=chunk.metadata))
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def process_uploaded_files(uploaded_files, progress_bar, status_text, chunking_strategy=None):
    docs = []
    total_files = len(uploaded_files)
    status_text.text(f"{get_text('progress_reading')} (0/{total_files})")
    
    # Get chunking strategy from session state or parameter
    strategy = chunking_strategy or st.session_state.get("chunking_strategy", DEFAULT_CHUNKING)
    
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
    
    # Use the selected chunking strategy
    if strategy == "semantic":
        try:
            # Load embeddings for semantic chunking
            embeddings = load_embeddings(st.session_state.device_choice)
            chunks = semantic_chunking(pages_list, embeddings, status_text)
        except Exception as e:
            # Fallback to fixed chunking if semantic fails
            st.warning(f"Semantisk segmentering misslyckades, använder standard. Fel: {e}" if st.session_state.get("lang","sv")=="sv" else f"Semantic chunking failed, using standard. Error: {e}")
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(pages_list)
    else:
        # Standard fixed-size chunking
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

# ---------- OPTIMIZED SUMMARIZATION WITH BATCH PROCESSING, MAPREDUCE & CACHING ----------
import hashlib

def generate_doc_hash(docs):
    """Generate a unique hash for the document set to use as cache key."""
    content = "".join([doc.page_content for doc in docs])
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_summary(doc_hash, focus, style, target_words):
    """Retrieve cached summary if available."""
    cache_key = f"{doc_hash}_{focus}_{style}_{target_words}"
    return st.session_state.get("summary_cache", {}).get(cache_key)

def set_cached_summary(doc_hash, focus, style, target_words, summary):
    """Store summary in cache."""
    cache_key = f"{doc_hash}_{focus}_{style}_{target_words}"
    if "summary_cache" not in st.session_state:
        st.session_state.summary_cache = {}
    st.session_state.summary_cache[cache_key] = summary

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
            
            # ===== STEP 1: BATCH CHUNKS TOGETHER =====
            # Group multiple pages into batches to reduce LLM calls
            BATCH_SIZE = 5  # Process 5 chunks at a time
            batches = [self.docs[i:i+BATCH_SIZE] for i in range(0, len(self.docs), BATCH_SIZE)]
            
            if self.lang == "sv":
                # Map phase: summarize each batch
                map_template = """Du är en undersökande journalist. Skapa en detaljerad sammanfattning av följande textavsnitt med fokus på: {focus}.
Inkludera viktiga fakta, namn, datum och sidhänvisningar.
Textavsnitt: {text}

Detaljerad sammanfattning av detta avsnitt:"""
                
                # Reduce phase: combine batch summaries
                reduce_template = """Du är en journalist och redaktör. Kombinera följande separata sammanfattningar till en sammanhängande helhet på ungefär {target_words} ord.
Fokus: {focus}. Stil: {style}.
Bevara viktiga fakta, namn, datum och källhänvisningar.

Sammanfattningar att kombinera:
{existing_summaries}

Kombinerad sammanfattning:"""
                
                final_prompt = f"""Du är en journalist och redaktör. Förfina följande sammanfattning till en slutlig, polerad version på ungefär {target_words} ord.
Stil: {{style}}. Fokus: {{focus}}.
Inkludera källhänvisningar [Källa: filnamn, sida X] där möjligt.

Sammanfattning att förfina:
{{current_summary}}

Slutlig sammanfattning:"""
            else:
                # English templates
                map_template = """You are an investigative journalist. Create a detailed summary of the following text passage, focusing on: {focus}.
Include key facts, names, dates, and page references.
Text passage: {text}

Detailed summary of this passage:"""
                
                reduce_template = """You are a journalist and editor. Combine the following separate summaries into a coherent whole of approximately {target_words} words.
Focus: {focus}. Style: {style}.
Preserve important facts, names, dates, and source citations.

Summaries to combine:
{existing_summaries}

Combined summary:"""
                
                final_prompt = """You are a journalist and editor. Refine the following summary into a final, polished version of approximately {target_words} words.
Style: {style}. Focus: {focus}.
Include citations [Source: filename, page X] where possible.

Summary to refine:
{current_summary}

Final summary:"""

            # ===== STEP 2: MAP PHASE - Process batches in parallel =====
            batch_summaries = []
            total_batches = len(batches)
            
            for batch_idx, batch in enumerate(batches):
                # Calculate percentage based on batch progress
                percentage = int((batch_idx + 1) / total_batches * 100)
                if self.progress_callback:
                    self.progress_callback("processing", batch_idx + 1, total_batches, percentage,
                        f"Sammanfattar avsnitt {batch_idx + 1} av {total_batches}..." if self.lang == "sv" else 
                        f"Summarizing section {batch_idx + 1} of {total_batches}...")
                
                # Combine text from all chunks in this batch
                batch_text = ""
                for doc in batch:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    batch_text += f"[Från {source}, sida {page}]: {doc.page_content}\n\n" if self.lang == "sv" else f"[From {source}, page {page}]: {doc.page_content}\n\n"
                
                try:
                    prompt = map_template.format(focus=self.focus, text=batch_text)
                    summary = self.llm.invoke(prompt).content
                    batch_summaries.append(summary)
                except Exception as e:
                    self.error = str(e)
                    return
            
            # ===== STEP 3: REDUCE PHASE - Combine batch summaries =====
            if len(batch_summaries) == 1:
                combined_summary = batch_summaries[0]
            else:
                # First reduce: combine all batch summaries
                combined = batch_summaries[0]
                for i in range(1, len(batch_summaries)):
                    try:
                        prompt = reduce_template.format(
                            existing_summaries=combined,
                            target_words=target_words,
                            focus=self.focus,
                            style=self.style
                        )
                        combined = self.llm.invoke(prompt).content
                    except Exception as e:
                        self.error = str(e)
                        return
                combined_summary = combined
            
            # ===== STEP 4: Final polish =====
            try:
                final_prompt_text = final_prompt.format(
                    current_summary=combined_summary,
                    target_words=target_words,
                    style=self.style,
                    focus=self.focus
                )
                final_summary = self.llm.invoke(final_prompt_text).content
            except Exception as e:
                self.error = str(e)
                return
            
            self.result = final_summary
        except Exception as e:
            self.error = str(e)

def start_summary(docs, target_pages, focus, style, words_per_page, lang):
    # Store parameters for caching
    st.session_state.last_target_words = target_pages * words_per_page
    st.session_state.last_focus = focus
    st.session_state.last_style = style
    
    # Check cache first
    doc_hash = generate_doc_hash(docs)
    target_words = target_pages * words_per_page
    
    cached = get_cached_summary(doc_hash, focus, style, target_words)
    if cached:
        st.session_state.last_summary = cached
        st.session_state.summary_result = cached
        st.success("📋 Hämtade från cache!" if lang == "sv" else "📋 Retrieved from cache!")
        return
    
    # Load LLM with current device choice
    device_choice = st.session_state.device_choice
    llm = load_llm(device_choice)
    thread = SummaryThread(docs, llm, target_pages, focus, style, words_per_page, lang, update_summary_progress)
    st.session_state.summary_thread = thread
    st.session_state.summary_progress = 0
    st.session_state.summary_total = len(docs)
    st.session_state.summary_running = True
    st.session_state.summary_result = None
    st.session_state.summary_error = None
    thread.start()

def update_summary_progress(stage, current, total, percentage, message):
    """Enhanced progress callback with detailed stage information."""
    # Initialize session state variables if they don't exist (for background thread safety)
    if "summary_stage" not in st.session_state:
        st.session_state.summary_stage = stage
    else:
        st.session_state.summary_stage = stage
    
    if "summary_current_batch" not in st.session_state:
        st.session_state.summary_current_batch = current
    else:
        st.session_state.summary_current_batch = current
        
    if "summary_total_batches" not in st.session_state:
        st.session_state.summary_total_batches = total
    else:
        st.session_state.summary_total_batches = total
        
    if "summary_percentage" not in st.session_state:
        st.session_state.summary_percentage = percentage
    else:
        st.session_state.summary_percentage = percentage
    
    # Add to log if message changed
    if message:
        # Initialize log if it doesn't exist
        if "summary_stages_log" not in st.session_state:
            st.session_state.summary_stages_log = []
            
        log_entry = {
            "stage": stage,
            "message": message,
            "timestamp": time.time(),
            "percentage": percentage
        }
        if not st.session_state.summary_stages_log or st.session_state.summary_stages_log[-1]["message"] != message:
            st.session_state.summary_stages_log.append(log_entry)
    
        # Keep log manageable - max 50 entries
        if len(st.session_state.summary_stages_log) > 50:
            st.session_state.summary_stages_log = st.session_state.summary_stages_log[-50:]

def check_summary_status():
    if st.session_state.get("summary_running", False):
        thread = st.session_state.get("summary_thread")
        if thread and not thread.is_alive():
            if thread.error:
                st.session_state.summary_error = thread.error
            else:
                st.session_state.summary_result = thread.result
                st.session_state.last_summary = thread.result
                
                # Store in cache for future use
                if st.session_state.docs and thread.result:
                    doc_hash = generate_doc_hash(st.session_state.docs)
                    target_words = st.session_state.get("last_target_words", 1500)
                    focus = st.session_state.get("last_focus", "")
                    style = st.session_state.get("last_style", "neutral")
                    set_cached_summary(doc_hash, focus, style, target_words, thread.result)
                    
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
    # Use adaptive retrieval - analyze query complexity to determine k
    k = analyze_query_complexity(query)
    context_docs = retrieve_context(query, vectorstore, k=k) if vectorstore else []
    prompt = create_chat_prompt(history, context_docs, query, lang)
    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        response = f"Fel vid generering av svar: {e}" if lang == "sv" else f"Error generating answer: {e}"
        return response, context_docs
    
    # Verify citations exist in retrieved context
    verified_response, citation_issues = verify_citations(response, context_docs, lang)
    
    # If there are citation issues, we can either regenerate or flag the issue
    # For now, we return the verified response with any warnings
    return verified_response, context_docs


# ---------- Adaptive Retrieval: Query Complexity Analysis ----------
def analyze_query_complexity(query: str) -> int:
    """
    Analyze query complexity and return optimal k value for retrieval.
    
    Simple queries: k=3-4
    Medium complexity: k=5-7
    Complex analytical: k=8-10
    """
    query_lower = query.lower()
    
    # Complexity indicators
    complexity_score = 0
    
    # Length-based complexity (longer queries often need more context)
    word_count = len(query.split())
    if word_count > 15:
        complexity_score += 2
    elif word_count > 8:
        complexity_score += 1
    
    # Analytical keywords indicate complex queries
    analytical_keywords = [
        'compare', 'analyze', 'explain', 'why', 'how', 'difference',
        'relationship', 'impact', 'effect', 'cause', 'result', 'consequence',
        'jämför', 'analysera', 'förklara', 'varför', 'hur', 'skillnad',
        'relation', 'påverkan', 'effekt', 'orsak', 'resultat', 'konsekvens'
    ]
    for keyword in analytical_keywords:
        if keyword in query_lower:
            complexity_score += 2
    
    # Multiple entities/concepts (question words)
    question_indicators = ['who', 'what', 'when', 'where', 'which', 'whom', 'whose',
                           'vem', 'vad', 'när', 'var', 'vilken', 'vilket', 'vilka']
    for indicator in question_indicators:
        if indicator in query_lower:
            complexity_score += 1
    
    # Comparison queries need more context
    if any(word in query_lower for word in ['vs', 'versus', 'compared', 'eller', 'jämfört']):
        complexity_score += 2
    
    # Map complexity score to k value
    if complexity_score <= 2:
        return 3  # Simple query
    elif complexity_score <= 5:
        return 5  # Medium complexity
    else:
        return 8  # Complex analytical query


# ---------- Citation Verification ----------
def extract_citations_from_response(response: str) -> list:
    """
    Extract citations from the LLM response.
    Looks for patterns like [Source: filename, page X] or [Källa: filename, sida X]
    """
    citations = []
    
    # Pattern for English: [Source: filename, page X]
    en_pattern = r'\[Source:\s*([^,\]]+),\s*page\s*(\d+)\]'
    # Pattern for Swedish: [Källa: filename, sida X]
    sv_pattern = r'\[Källa:\s*([^,\]]+),\s*sida\s*(\d+)\]'
    
    # Also match bare citations like (source, page 5) or [source, p.5]
    bare_pattern = r'\[?([A-Za-z0-9_\-\.]+)[,\s]+(?:page|p\.?|sida|s\.)\s*(\d+)\]?'
    
    import re
    for match in re.finditer(en_pattern, response, re.IGNORECASE):
        citations.append({
            'source': match.group(1).strip(),
            'page': match.group(2).strip()
        })
    
    for match in re.finditer(sv_pattern, response, re.IGNORECASE):
        citations.append({
            'source': match.group(1).strip(),
            'page': match.group(2).strip()
        })
    
    for match in re.finditer(bare_pattern, response, re.IGNORECASE):
        citations.append({
            'source': match.group(1).strip(),
            'page': match.group(2).strip()
        })
    
    return citations


def verify_citations(response: str, context_docs: list, lang: str) -> tuple:
    """
    Verify that citations in the response actually exist in the retrieved context.
    Returns (verified_response, issues)
    """
    if not context_docs:
        return response, ["No context available to verify citations"]
    
    # Extract citations from response
    citations = extract_citations_from_response(response)
    
    if not citations:
        # No citations found - this is fine, just return as-is
        return response, []
    
    # Build a set of available sources from context
    available_sources = {}
    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        if source not in available_sources:
            available_sources[source] = set()
        available_sources[source].add(str(page))
    
    # Verify each citation
    issues = []
    verified_citations = []
    
    for citation in citations:
        source = citation['source']
        page = citation['page']
        
        # Check if source exists in context
        source_found = False
        for avail_source in available_sources:
            # Partial match (e.g., "report.pdf" matches "report_final.pdf")
            if source.lower() in avail_source.lower() or avail_source.lower() in source.lower():
                source_found = True
                # Check if page exists (optional - some sources might not have page numbers)
                if page in available_sources[avail_source] or page == "?":
                    verified_citations.append(citation)
                else:
                    # Page not found but source exists - might still be valid
                    verified_citations.append(citation)
                break
        
        if not source_found:
            issues.append(f"Citation refers to '{source}' which was not found in retrieved context")
    
    # If there are issues, add a warning to the response
    if issues:
        warning_msg = ""
        if lang == "sv":
            warning_msg = "\n\n⚠️ *Obs: Vissa källhänvisningar kunde inte verifieras i dokumentkontexten.*"
        else:
            warning_msg = "\n\n⚠️ *Note: Some citations could not be verified in the document context.*"
        
        response = response + warning_msg
    
    return response, issues

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
# UI Custom CSS (unchanged)
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
        "device_choice": "auto",  # default
        "llm_model": DEFAULT_LLM_MODEL,  # Quantized model default
        "chunking_strategy": DEFAULT_CHUNKING,  # Semantic chunking default
        "processing": False,
        "summary_running": False,
        "summary_thread": None,
        "summary_progress": 0,
        "summary_total": 0,
        "summary_result": None,
        "summary_error": None,
        # New UX improvements - detailed progress tracking
        "summary_stage": "idle",  # idle, reading, processing, combining, polishing, complete, error, cancelled
        "summary_stages_log": [],  # List of all stage changes with timestamps
        "summary_current_batch": 0,
        "summary_total_batches": 0,
        "summary_percentage": 0,
        "summary_cancel_requested": False,
        "summary_start_time": None,
        "summary_stage_duration": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Check if summary thread finished
    check_summary_status()

    # Sidebar
    with st.sidebar:
        st.markdown(f"## {get_text('title')}")
        current_device = resolve_device(st.session_state.device_choice)
        st.caption(f"Körs på **{current_device.upper()}**")
        st.divider()

        # Language selector
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

        # Device selector with explanation
        st.markdown(f"### {get_text('device')}")
        st.info(get_text("device_info"))
        device_choice = st.selectbox(
            get_text("device"),
            options=["auto", "cuda", "cpu"],
            format_func=lambda x: get_text(f"device_{x}"),
            key="device_selector",
            help=get_text("ollama_gpu_note")
        )
        if device_choice != st.session_state.device_choice:
            st.session_state.device_choice = device_choice
            st.rerun()  # Rerun to reload resources with new device

        # Show current effective device
        effective_device = resolve_device(st.session_state.device_choice)
        st.caption(get_text("device_current").format(effective_device.upper()))
        
        st.divider()
        
        # Embedding model selector (improves semantic understanding)
        # User-friendly selector with explanations for non-technical users
        st.markdown(f"### {get_text('embedding_model')}")
        
        # Create a more user-friendly selector with descriptions
        def get_embedding_display_text(key):
            model_info = EMBEDDING_MODELS.get(key, {})
            if isinstance(model_info, dict):
                current_lang = st.session_state.get("lang", "sv")
                display_name = model_info.get(f"display_name_en" if current_lang == "en" else "display_name", key)
                description = model_info.get(f"description_en" if current_lang == "en" else "description", "")
                speed = model_info.get("speed", "")
                quality = model_info.get("quality", "")
                return f"{display_name}\n{speed} {quality}\n{description}"
            return key
        
        # Get current selection
        current_embedding = st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        
        # Display radio buttons with detailed explanations for better UX
        st.markdown("**Välj modell:**" if st.session_state.get("lang", "sv") == "sv" else "**Select model:**")
        
        for model_key, model_info in EMBEDDING_MODELS.items():
            current_lang = st.session_state.get("lang", "sv")
            display_name = model_info.get(f"display_name_en" if current_lang == "en" else "display_name", model_key)
            description = model_info.get(f"description_en" if current_lang == "en" else "description", "")
            speed = model_info.get("speed", "")
            quality = model_info.get("quality", "")
            
            # Create a styled radio option
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    is_selected = st.radio(
                        "select_embedding_model",
                        options=[model_key],
                        format_func=lambda x: "●" if x == current_embedding else "○",
                        key=f"radio_{model_key}",
                        label_visibility="collapsed"
                    )
                with col2:
                    if model_key == current_embedding:
                        st.markdown(f"**{display_name}** {speed} {quality}")
                    else:
                        st.markdown(f"{display_name} {speed} {quality}")
                    st.caption(description)
                st.markdown("---")
        
        # Use selectbox for actual selection (hidden style)
        embedding_model = st.selectbox(
            get_text("embedding_model"),
            options=list(EMBEDDING_MODELS.keys()),
            format_func=get_embedding_display_text,
            key="embedding_model_selector",
            label_visibility="collapsed",
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        )
        if embedding_model != st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL):
            st.session_state.embedding_model = embedding_model
            st.rerun()  # Rerun to reload resources with new embedding model

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
                    status.text(get_text("progress_indexing"))
                    embeddings = load_embeddings(st.session_state.device_choice)
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
                            embeddings = load_embeddings(st.session_state.device_choice)
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
                llm = load_llm(st.session_state.device_choice)
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
            # Enhanced progress UI with detailed stages
            stage = st.session_state.get("summary_stage", "processing")
            percentage = st.session_state.get("summary_percentage", 0)
            current_batch = st.session_state.get("summary_current_batch", 0)
            total_batches = st.session_state.get("summary_total_batches", 0)
            
            # Stage indicator pills
            stage_icons = {
                "idle": "⚪",
                "initializing": "🔄",
                "processing": "📝",
                "combining": "🔗",
                "polishing": "✨",
                "complete": "✅",
                "error": "❌",
                "cancelled": "⏹️"
            }
            current_icon = stage_icons.get(stage, "⏳")
            
            # Stage labels
            stage_labels = {
                "idle": "Väntar" if lang=="sv" else "Idle",
                "initializing": "Initierar" if lang=="sv" else "Initializing",
                "processing": "Sammanfattar avsnitt" if lang=="sv" else "Processing sections",
                "combining": "Kombinerar sammanfattningar" if lang=="sv" else "Combining summaries",
                "polishing": "Färdigställer" if lang=="sv" else "Finalizing",
                "complete": "Klart" if lang=="sv" else "Complete",
                "error": "Fel" if lang=="sv" else "Error",
                "cancelled": "Avbrutet" if lang=="sv" else "Cancelled"
            }
            
            # Show current stage with icon
            st.markdown(f"""
            <div class="info-box">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 24px;">{current_icon}</span>
                    <span style="font-weight: 600; font-size: 16px;">{stage_labels.get(stage, stage)}</span>
                    <span style="margin-left: auto; font-weight: bold; color: #3b82f6;">{percentage:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Overall progress bar
            st.progress(percentage / 100.0)
            
            # Stage progress indicators (showing all stages)
            cols = st.columns(4)
            stages_order = ["processing", "combining", "polishing", "complete"]
            for i, s in enumerate(stages_order):
                with cols[i]:
                    is_active = (stage == s)
                    is_complete = (stages_order.index(stage) > i) if stage in stages_order else False
                    if is_complete:
                        st.markdown(f"✅ **{stage_labels.get(s, s)}**")
                    elif is_active:
                        st.markdown(f"🔄 **{stage_labels.get(s, s)}**")
                    else:
                        st.markdown(f"⚪ {stage_labels.get(s, s)}")
            
            # Show current batch info if processing
            if stage == "processing" and total_batches > 0:
                st.caption(f"{'Batch' if lang=='sv' else 'Batch'} {current_batch} / {total_batches}")
            
            # Expandable log
            with st.expander("📋 " + ("Detaljerad logg" if lang=="sv" else "Detailed Log")):
                log = st.session_state.get("summary_stages_log", [])
                for entry in log[-10:]:  # Show last 10 entries
                    st.caption(f"{entry.get('percentage', 0):.0f}% - {entry.get('message', '')}")
            
            # Cancel button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("⏹️ " + ("Avbryt" if lang=="sv" else "Cancel"), key="cancel_summary"):
                    st.session_state.summary_cancel_requested = True
                    if st.session_state.get("summary_thread"):
                        st.session_state.summary_thread.stop()
                    st.session_state.summary_running = False
                    st.session_state.summary_stage = "cancelled"
                    st.rerun()
                    
        elif st.session_state.summary_error:
            error_msg = st.session_state.summary_error
            if error_msg == "cancelled":
                st.markdown(f'<div class="warning-box">⚠️ {"Åtgärden avbröts av användaren." if lang=="sv" else "Operation cancelled by user."}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">❌ {"Fel: " if lang=="sv" else "Error: "}{error_msg}</div>', unsafe_allow_html=True)
        elif st.session_state.summary_result:
            st.markdown("### " + ("Sammanfattning" if lang=="sv" else "Summary"))
            st.write(st.session_state.summary_result)
            st.session_state.last_summary = st.session_state.summary_result
            # Show success message with duration if available
            if st.session_state.get("summary_stages_log"):
                # Find completion entry
                for entry in reversed(st.session_state.summary_stages_log):
                    if "complete" in entry.get("message", "").lower() or "slutfört" in entry.get("message", "").lower():
                        st.success(entry.get("message", get_text("success_summary")))
                        break
                else:
                    st.success(get_text("success_summary"))

        # Generate button
        if st.button(get_text("summarize_btn"), disabled=st.session_state.processing or st.session_state.summary_running or not st.session_state.docs, key="summarize_btn_main"):
            if not st.session_state.docs:
                st.error(get_text("error_no_docs"))
            else:
                # Reset progress state before starting
                st.session_state.summary_stage = "initializing"
                st.session_state.summary_stages_log = []
                st.session_state.summary_percentage = 0
                st.session_state.summary_current_batch = 0
                st.session_state.summary_total_batches = 0
                st.session_state.summary_cancel_requested = False
                start_summary(st.session_state.docs, target_pages, focus, style, words_per_page, st.session_state.lang)
                st.rerun()

    # Analysis tools
    with st.expander(get_text("analysis_title"), expanded=False):
        if not st.session_state.docs:
            st.warning("Ladda upp dokument först." if lang=="sv" else "Please upload documents first.")
        else:
            ner_pipeline = load_ner_pipeline(st.session_state.device_choice)
            sent_pipeline = load_sentiment_pipeline(st.session_state.device_choice)
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
                    llm = load_llm(st.session_state.device_choice)
                    critique = bias_check(text, llm, st.session_state.lang)
                    st.markdown("### Partiskhetsgranskning" if lang=="sv" else "### Bias Critique")
                    st.write(critique)

            target_lang = st.selectbox(get_text("translate"), ["Svenska", "English", "Spanish", "French", "German"] if lang=="sv" else ["Swedish", "English", "Spanish", "French", "German"])
            if st.button("Översätt" if lang=="sv" else "Translate", disabled=st.session_state.processing):
                with st.spinner("Översätter..." if lang=="sv" else "Translating..."):
                    llm = load_llm(st.session_state.device_choice)
                    translated = translate_text(text, target_lang, llm, st.session_state.lang)
                    st.markdown(f"### Översättning ({target_lang})" if lang=="sv" else f"### Translation ({target_lang})")
                    st.write(translated)

    # Footer
    st.divider()
    st.caption("Insikt – 100% lokalt, privat och gratis." if lang=="sv" else "Insikt – 100% local, private, and free.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Ett kritiskt fel uppstod. Starta om appen." if st.session_state.get("lang","sv")=="sv" else "A critical error occurred. Please restart the app.")
        st.exception(e)