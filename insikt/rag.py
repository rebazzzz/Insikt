from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


def analyze_query_complexity(query: str) -> int:
    query_lower = query.lower()
    score = 0
    word_count = len(query.split())
    if word_count > 15:
        score += 2
    elif word_count > 8:
        score += 1

    analytical_keywords = [
        "compare", "analyze", "explain", "why", "how", "difference",
        "relationship", "impact", "effect", "cause", "result", "consequence",
        "jämför", "analysera", "förklara", "varför", "hur", "skillnad",
        "relation", "påverkan", "effekt", "orsak", "resultat", "konsekvens",
    ]
    for keyword in analytical_keywords:
        if keyword in query_lower:
            score += 2

    if any(word in query_lower for word in ["vs", "versus", "compared", "jämfört"]):
        score += 2

    if score <= 2:
        return 4
    if score <= 5:
        return 6
    return 9


def _normalize_tokens(text: str) -> set:
    return set(re.findall(r"[A-Za-zÅÄÖåäö0-9]{3,}", text.lower()))


def _doc_matches_filter(doc: Document, source_filter: Optional[Sequence[str]]) -> bool:
    if not source_filter:
        return True
    source = str(doc.metadata.get("source", ""))
    return any(selected == source for selected in source_filter)


def rerank_documents(query: str, docs: Sequence[Document], limit: int = 6) -> List[Document]:
    query_tokens = _normalize_tokens(query)
    scored = []
    for index, doc in enumerate(docs):
        doc_tokens = _normalize_tokens(doc.page_content[:2500])
        overlap = len(query_tokens & doc_tokens)
        density = overlap / max(len(query_tokens), 1)
        lexical_hits = sum(1 for token in query_tokens if token in doc.page_content.lower())
        position_bonus = max(0.0, 1.0 - (index * 0.05))
        score = (density * 4.0) + (lexical_hits * 0.2) + position_bonus
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def retrieve_context(query: str, vectorstore, k: int = 7, source_filter: Optional[Sequence[str]] = None) -> List[Document]:
    if not vectorstore:
        return []
    candidates = vectorstore.similarity_search(query, k=max(k * 3, 9))
    filtered = [doc for doc in candidates if _doc_matches_filter(doc, source_filter)]
    if not filtered and source_filter:
        filtered = candidates
    return rerank_documents(query, filtered, limit=k)


def build_context_string(context_docs: Sequence[Document], lang: str) -> str:
    if not context_docs:
        return "Inga dokument tillgängliga eller ingen relevant information hittades." if lang == "sv" else "No documents available or no relevant information found."
    lines = []
    for doc in context_docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        label = f"[Källa: {src}, sida {page}]" if lang == "sv" else f"[Source: {src}, page {page}]"
        lines.append(f"{label}: {doc.page_content}")
    return "\n\n".join(lines)


def build_history_string(history: Sequence, lang: str) -> str:
    lines = []
    for msg in history[-10:]:
        if lang == "sv":
            role = "Användare" if isinstance(msg, HumanMessage) else "Assistent"
        else:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def create_chat_prompt(history, context_docs, query, lang):
    if lang == "sv":
        system_template = """
Du är en hjälpsam och ärlig AI-assistent. Svara bara med stöd i dokumentkontexten när den finns.
När du använder information från dokumenten måste du ange källa som [Källa: filnamn, sida X].
Om dokumenten inte räcker ska du säga det tydligt och skilja på dokumentstöd och allmän kunskap.
Om flera källor säger olika saker ska du säga det.

Kontext från dokument:
{context}

Tidigare konversation:
{history}

Användare: {query}
Assistent:"""
    else:
        system_template = """
You are a helpful and honest AI assistant. Answer with support from the document context whenever it exists.
When you use document information, cite it as [Source: filename, page X].
If the documents are insufficient, say so clearly and separate document-grounded claims from general knowledge.
If sources conflict, say that explicitly.

Context from documents:
{context}

Conversation history:
{history}

User: {query}
Assistant:"""
    return system_template.format(context=build_context_string(context_docs, lang), history=build_history_string(history, lang), query=query)


def extract_citations_from_response(response: str) -> list:
    citations = []
    en_pattern = r"\[Source:\s*([^,\]]+),\s*page\s*(\d+)\]"
    sv_pattern = r"\[Källa:\s*([^,\]]+),\s*sida\s*(\d+)\]"
    bare_pattern = r"\[?([A-Za-z0-9_\-\.]+)[,\s]+(?:page|p\.?|sida|s\.)\s*(\d+)\]?"
    for pattern in [en_pattern, sv_pattern, bare_pattern]:
        for match in re.finditer(pattern, response, re.IGNORECASE):
            citations.append({"source": match.group(1).strip(), "page": match.group(2).strip()})
    return citations


def verify_citations(response: str, context_docs: Sequence[Document], lang: str) -> tuple:
    if not context_docs:
        return response, ["No context available to verify citations"]
    citations = extract_citations_from_response(response)
    if not citations:
        warning = "\n\nObs: Svaret saknar tydliga källhänvisningar." if lang == "sv" else "\n\nNote: The answer is missing explicit citations."
        return response + warning, ["missing_citations"]

    available_sources = {}
    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")
        page = str(doc.metadata.get("page", "?"))
        available_sources.setdefault(source, set()).add(page)

    issues = []
    for citation in citations:
        source = citation["source"]
        page = citation["page"]
        if source not in available_sources:
            issues.append(f"unknown_source:{source}")
            continue
        if page not in available_sources[source]:
            issues.append(f"unknown_page:{source}:{page}")

    if issues:
        warning = "\n\nObs: Vissa källhänvisningar kunde inte verifieras i dokumentkontexten." if lang == "sv" else "\n\nNote: Some citations could not be verified in the document context."
        response = response + warning
    return response, issues


def grounding_check(response: str, context_docs: Sequence[Document], lang: str) -> Tuple[str, list]:
    if not context_docs:
        return response, []
    response_tokens = _normalize_tokens(response)
    context_tokens = set()
    for doc in context_docs:
        context_tokens.update(_normalize_tokens(doc.page_content[:2000]))
    overlap_ratio = len(response_tokens & context_tokens) / max(len(response_tokens), 1)
    if overlap_ratio >= 0.12:
        return response, []
    warning = "\n\nObs: Delar av svaret verkar svagt förankrade i det hämtade underlaget." if lang == "sv" else "\n\nNote: Parts of the answer appear weakly grounded in the retrieved context."
    return response + warning, ["weak_grounding"]


def _build_sources_list(docs, lang, limit=15):
    items = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{src}:{page}"
        if key in seen:
            continue
        seen.add(key)
        items.append(f"{src}, sida {page}" if lang == "sv" else f"{src}, page {page}")
        if len(items) >= limit:
            break
    return "\n".join(items)


def _has_citations(text, lang):
    return bool(re.search(r"\[Källa:\s*[^\]]+\]", text)) if lang == "sv" else bool(re.search(r"\[Source:\s*[^\]]+\]", text))


def _citation_fix(text, lang, llm, sources_list):
    if not sources_list:
        return text
    if lang == "sv":
        prompt = f"""Du är en redaktör. Lägg till tydliga källhänvisningar i texten.
Använd endast denna lista över källor:
{sources_list}

Text:
{text}

Uppdaterad text med källor:"""
    else:
        prompt = f"""You are an editor. Add clear citations to the text.
Use only this list of sources:
{sources_list}

Text:
{text}

Updated text with citations:"""
    try:
        return llm.invoke(prompt).content
    except Exception:
        return text


def create_writing_prompt(brief, role_label, format_label, tone_label, length_words, lang, context_docs):
    sources = build_context_string(context_docs, lang)
    if lang == "sv":
        return f"""Du är en {role_label}. Skapa ett utkast i formatet {format_label}.
Ton: {tone_label}. Mållängd: cirka {length_words} ord.

Uppdrag:
{brief}

När du använder fakta från dokument, inkludera källhänvisningar [Källa: filnamn, sida X].
Om fakta saknas, markera antaganden tydligt.

Källor:
{sources}

Leverera:
1) Kort disposition
2) Utkast eller manus med tydlig struktur
3) Lista över sådant som behöver verifieras"""
    return f"""You are a {role_label}. Create a draft in the format {format_label}.
Tone: {tone_label}. Target length: about {length_words} words.

Assignment:
{brief}

When using facts from documents, include citations [Source: filename, page X].
If facts are missing, clearly mark assumptions.

Sources:
{sources}

Deliver:
1) Short outline
2) Draft or script with clear structure
3) Verification needs list"""


def chat_with_docs(query, history, vectorstore, llm, lang, source_filter=None):
    context_docs = retrieve_context(query, vectorstore, k=analyze_query_complexity(query), source_filter=source_filter)
    prompt = create_chat_prompt(history, context_docs, query, lang)
    try:
        response = llm.invoke(prompt).content
    except Exception as exc:
        return (f"Fel vid generering av svar: {exc}" if lang == "sv" else f"Error generating answer: {exc}"), context_docs, ["generation_error"]
    response, citation_issues = verify_citations(response, context_docs, lang)
    response, grounding_issues = grounding_check(response, context_docs, lang)
    return response, context_docs, citation_issues + grounding_issues


def generate_writing(brief, role_label, format_label, tone_label, length_words, lang, vectorstore, llm, use_sources=True, use_pipeline=False, source_filter=None):
    context_docs = retrieve_context(brief, vectorstore, k=6, source_filter=source_filter) if vectorstore and use_sources else []
    prompt = create_writing_prompt(brief, role_label, format_label, tone_label, length_words, lang, context_docs)
    sources_list = _build_sources_list(context_docs, lang)
    try:
        if use_pipeline and "document" in format_label.lower():
            outline_prompt = prompt + ("\n\nSteg 1: Skapa en tydlig disposition." if lang == "sv" else "\n\nStep 1: Create a clear outline.")
            outline = llm.invoke(outline_prompt).content
            scene_prompt = ("\n\nSteg 2: Skapa en scenlista baserat på dispositionen:\n" if lang == "sv" else "\n\nStep 2: Create a scene list based on the outline:\n") + outline
            scenes = llm.invoke(scene_prompt).content
            script_prompt = ("\n\nSteg 3: Skriv slutligt manus baserat på scenlistan:\n" if lang == "sv" else "\n\nStep 3: Write the final script based on the scene list:\n") + scenes
            response = f"{outline}\n\n{scenes}\n\n{llm.invoke(script_prompt).content}"
        else:
            response = llm.invoke(prompt).content
    except Exception as exc:
        return (f"Fel vid generering: {exc}" if lang == "sv" else f"Error generating text: {exc}"), context_docs

    if use_sources and context_docs and not _has_citations(response, lang):
        response = _citation_fix(response, lang, llm, sources_list)
    return response, context_docs
