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
        "jamfor", "jämför", "analysera", "forklara", "förklara", "varfor", "varför",
        "hur", "skillnad", "relation", "paverkan", "påverkan", "effekt", "orsak",
        "resultat", "konsekvens",
    ]
    for keyword in analytical_keywords:
        if keyword in query_lower:
            score += 2

    if any(word in query_lower for word in ["vs", "versus", "compared", "jamfort", "jämfört"]):
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


def _build_chat_system_prompt(lang: str, has_context: bool) -> str:
    if lang == "sv":
        if has_context:
            return """
Du är en hjälpsam, varm och ärlig AI-assistent.
Du får gärna låta mänsklig och samtalsvänlig, men du får aldrig hitta på fakta.

Regler:
- Använd dokumentkontexten som primär kunskapskälla.
- När du använder information från dokumenten måste du citera den som [Källa: filnamn, sida X].
- Om dokumenten inte räcker ska du säga det tydligt.
- Om frågan inte kan besvaras säkert från materialet ska du säga vad som saknas i stället för att gissa.
- Om användaren småpratar eller ber om kreativ hjälp får du svara naturligt utan att låtsas att det kommer från dokument.
- Om du använder allmän kunskap, märk det tydligt som allmän vägledning och håll dig försiktig.
- Om källor motsäger varandra ska du säga det uttryckligen.
""".strip()
        return """
Du är en hjälpsam, varm och ärlig AI-assistent.
Det finns inga uppladdade dokument att luta sig mot just nu.

Regler:
- Var naturlig och samtalsvänlig.
- Hitta inte på fakta. Om du inte vet, säg det.
- För faktafrågor ska du vara tydlig med att svaret inte är dokumentverifierat.
- För småprat, brainstorming, skrivhjälp och allmän vägledning kan du svara direkt, men utan att låtsas vara säker på specifika fakta du inte kan belägga.
""".strip()
    if has_context:
        return """
You are a helpful, warm, and honest AI assistant.
Sound human and conversational, but never invent facts.

Rules:
- Use the document context as your primary knowledge base.
- When you use document information, cite it as [Source: filename, page X].
- If the documents are insufficient, say so clearly.
- If the question cannot be answered safely from the material, explain what is missing instead of guessing.
- If the user is just chatting or asking for creative help, respond naturally without pretending it came from the documents.
- If you use general knowledge, label it clearly as general guidance and stay cautious.
- If sources conflict, say that explicitly.
""".strip()
    return """
You are a helpful, warm, and honest AI assistant.
There are no uploaded documents available right now.

Rules:
- Be natural and conversational.
- Do not make up facts. If you do not know, say so.
- For factual questions, be clear that the answer is not document-verified.
- For small talk, brainstorming, writing help, and general guidance, respond directly without pretending certainty about specific facts you cannot support.
""".strip()


def create_chat_prompt(history, context_docs, query, lang):
    system_prompt = _build_chat_system_prompt(lang, bool(context_docs))
    if lang == "sv":
        return f"""
{system_prompt}

Kontext från dokument:
{build_context_string(context_docs, lang)}

Tidigare konversation:
{build_history_string(history, lang)}

Användare: {query}
Assistent:""".strip()
    return f"""
{system_prompt}

Context from documents:
{build_context_string(context_docs, lang)}

Conversation history:
{build_history_string(history, lang)}

User: {query}
Assistant:""".strip()


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
        return response, ["no_knowledge_base"]
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


def assess_answer_confidence(response: str, context_docs: Sequence[Document], issues: Sequence[str] | None, lang: str) -> dict:
    issues = list(issues or [])
    citations = extract_citations_from_response(response)
    unique_citations = {(item["source"], item["page"]) for item in citations}
    source_count = len({doc.metadata.get("source", "Unknown") for doc in context_docs}) if context_docs else 0
    severe_issue_prefixes = ("generation_error", "missing_citations", "unknown_source", "unknown_page")
    has_severe_issue = any(issue.startswith(severe_issue_prefixes) for issue in issues)
    has_grounding_issue = any(issue.startswith("weak_grounding") for issue in issues)

    if not context_docs:
        score = 12 if not has_severe_issue else 5
    else:
        score = 20
        score += min(len(unique_citations), 4) * 15
        score += min(source_count, 3) * 8
        if has_grounding_issue:
            score -= 20
        if has_severe_issue:
            score -= 35
        elif not unique_citations:
            score -= 25
        score = max(0, min(score, 98))

    if has_severe_issue or score < 40:
        return {
            "level": "needs_review",
            "label": "Behöver granskas" if lang == "sv" else "Needs review",
            "reason": "Svaret saknar tillräckligt tydligt eller verifierat dokumentstöd." if lang == "sv" else "The answer lacks sufficiently clear or verified document support.",
            "citation_count": len(unique_citations),
            "source_count": source_count,
            "issues": issues,
            "score": score,
        }
    if has_grounding_issue or len(unique_citations) <= 1 or source_count <= 1:
        return {
            "level": "partly_supported",
            "label": "Delvis styrkt" if lang == "sv" else "Partly supported",
            "reason": "Delar av svaret stöds, men underlaget är begränsat eller bör dubbelkollas." if lang == "sv" else "Parts of the answer are supported, but the evidence is limited or should be double-checked.",
            "citation_count": len(unique_citations),
            "source_count": source_count,
            "issues": issues,
            "score": score,
        }
    return {
        "level": "well_supported",
        "label": "Väl underbyggt" if lang == "sv" else "Well-supported",
        "reason": "Svaret har flera verifierbara hänvisningar och tydligt stöd i dokumenten." if lang == "sv" else "The answer has multiple verifiable references and clear support in the documents.",
        "citation_count": len(unique_citations),
        "source_count": source_count,
        "issues": issues,
        "score": score,
    }


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
    if context_docs:
        response, citation_issues = verify_citations(response, context_docs, lang)
        response, grounding_issues = grounding_check(response, context_docs, lang)
    else:
        citation_issues = ["no_knowledge_base"]
        grounding_issues = []
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
