from __future__ import annotations

import threading
from typing import Sequence

from langchain_ollama import ChatOllama

from .common import compute_text_hash
from .rag import _build_sources_list, _citation_fix, _has_citations


def generate_doc_hash(docs: Sequence) -> str:
    return compute_text_hash(doc.page_content or "" for doc in docs)


class SummaryThread(threading.Thread):
    def __init__(self, docs, llm_model, target_pages, focus, style, words_per_page, lang, use_refine, progress_callback):
        super().__init__()
        self.docs = docs
        self.llm_model = llm_model
        self.target_pages = target_pages
        self.focus = focus
        self.style = style
        self.words_per_page = words_per_page
        self.lang = lang
        self.use_refine = use_refine
        self.progress_callback = progress_callback
        self.result = None
        self.error = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _check_cancelled(self):
        if self._stop_event.is_set():
            self.error = "cancelled"
            return True
        return False

    def _build_batches(self, max_chars=12000, max_chunks=8):
        batches = []
        current = []
        current_chars = 0
        for doc in self.docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            label = f"[Från {source}, sida {page}]" if self.lang == "sv" else f"[From {source}, page {page}]"
            entry = f"{label}: {doc.page_content}\n\n"
            if current and (current_chars + len(entry) > max_chars or len(current) >= max_chunks):
                batches.append("".join(current))
                current = []
                current_chars = 0
            current.append(entry)
            current_chars += len(entry)
        if current:
            batches.append("".join(current))
        return batches

    def run(self):
        try:
            target_words = min(max(150, self.target_pages * self.words_per_page), 4000)
            llm = ChatOllama(model=self.llm_model, temperature=0.2, num_predict=min(max(int(target_words * 1.3), 512), 4096))
            batches = self._build_batches()

            if self.lang == "sv":
                map_template = """Du är en undersökande journalist. Skapa en detaljerad sammanfattning av följande textavsnitt med fokus på: {focus}.
Stil: {style}. Inkludera viktiga fakta, namn, datum och sidhänvisningar.
Textavsnitt: {text}

Detaljerad sammanfattning av detta avsnitt:"""
                reduce_template = """Du är en journalist och redaktör. Kombinera följande separata sammanfattningar till en sammanhängande helhet på ungefär {target_words} ord.
Fokus: {focus}. Stil: {style}.
Bevara viktiga fakta, namn, datum och källhänvisningar.

Sammanfattningar att kombinera:
{existing_summaries}

Kombinerad sammanfattning:"""
                final_prompt = """Du är en journalist och redaktör. Förfina följande sammanfattning till en slutlig, polerad version på ungefär {target_words} ord.
Stil: {style}. Fokus: {focus}.
Inkludera källhänvisningar [Källa: filnamn, sida X] där möjligt.

Sammanfattning att förfina:
{current_summary}

Slutlig sammanfattning:"""
            else:
                map_template = """You are an investigative journalist. Create a detailed summary of the following text passage, focusing on: {focus}.
Style: {style}. Include key facts, names, dates, and page references.
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

            batch_summaries = []
            total_batches = max(len(batches), 1)
            for batch_idx, batch in enumerate(batches, start=1):
                if self._check_cancelled():
                    return
                if self.progress_callback:
                    self.progress_callback("processing", batch_idx, total_batches, int(batch_idx / total_batches * 70), f"Batch {batch_idx}/{total_batches}")
                batch_summaries.append(llm.invoke(map_template.format(focus=self.focus, style=self.style, text=batch)).content)

            if self._check_cancelled():
                return

            combined = "\n\n".join(batch_summaries)
            if self.progress_callback:
                self.progress_callback("combining", total_batches, total_batches, 82, "Combining summaries")
            combined = llm.invoke(reduce_template.format(focus=self.focus, style=self.style, target_words=target_words, existing_summaries=combined)).content

            if self.use_refine:
                if self.progress_callback:
                    self.progress_callback("polishing", total_batches, total_batches, 95, "Finalizing")
                combined = llm.invoke(final_prompt.format(style=self.style, focus=self.focus, target_words=target_words, current_summary=combined)).content

            sources_list = _build_sources_list(self.docs, self.lang)
            combined = _citation_fix(combined, self.lang, llm, sources_list)
            if not _has_citations(combined, self.lang):
                combined += "\n\nObs: Källor kunde inte verifieras." if self.lang == "sv" else "\n\nNote: Citations could not be verified."
            self.result = combined
        except Exception as exc:
            self.error = str(exc)
