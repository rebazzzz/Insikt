from __future__ import annotations

import re

try:
    import yake
except ImportError:  # pragma: no cover - optional dependency fallback
    yake = None


def extract_entities(text, ner_pipeline):
    if not ner_pipeline:
        return {}
    results = ner_pipeline(text[:10000])
    entities = {"PERSON": set(), "ORG": set(), "LOC": set()}
    for ent in results:
        if ent["entity_group"] in entities:
            entities[ent["entity_group"]].add(ent["word"])
    return entities


def extract_timeline(docs):
    date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"
    timeline = []
    for doc in docs:
        text = doc.page_content
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            timeline.append({"date": match.group(), "context": text[start:end].replace("\n", " "), "source": doc.metadata.get("source", "Unknown"), "page": doc.metadata.get("page", "?")})
    timeline.sort(key=lambda item: item["date"])
    return timeline


def extract_quote_candidates(docs, max_quotes=30):
    quote_pattern = r"[\"“”'']([^\"“”\n]{20,280})[\"“”'']"
    candidates = []
    seen = set()
    for doc in docs:
        text = doc.page_content or ""
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        for match in re.finditer(quote_pattern, text):
            quote = re.sub(r"\s+", " ", match.group(1)).strip()
            key = (quote.lower(), source, str(page))
            if len(quote.split()) < 4 or key in seen:
                continue
            seen.add(key)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = re.sub(r"\s+", " ", text[start:end]).strip()
            candidates.append({
                "quote": quote,
                "source": source,
                "page": page,
                "context": context,
                "length": len(quote),
            })
            if len(candidates) >= max_quotes:
                return candidates
    candidates.sort(key=lambda item: (-len(item["quote"].split()), item["source"], str(item["page"])))
    return candidates[:max_quotes]


def extract_claim_check_items(text, lang="en", max_items=40):
    sentence_pattern = r"(?<=[.!?])\s+"
    citation_pattern = r"\[(?:Source|Källa):[^\]]+\]"
    speculative_markers = [
        "may", "might", "could", "appears", "reportedly", "allegedly", "suggests",
        "kan", "kanske", "möjligen", "uppges", "påstås", "verkar", "tyder på",
    ]
    items = []
    for raw_sentence in re.split(sentence_pattern, text or ""):
        sentence = raw_sentence.strip()
        if len(sentence.split()) < 6:
            continue
        has_citation = bool(re.search(citation_pattern, sentence, re.IGNORECASE))
        needs_review = not has_citation
        reasons = []
        if not has_citation:
            reasons.append("missing_citation")
        lowered = sentence.lower()
        if any(marker in lowered for marker in speculative_markers):
            needs_review = True
            reasons.append("speculative_language")
        if re.search(r"\b\d[\d,.:/-]*\b", sentence) and not has_citation:
            needs_review = True
            reasons.append("number_without_citation")
        items.append({
            "sentence": sentence,
            "needs_review": needs_review,
            "has_citation": has_citation,
            "reasons": reasons,
        })
        if len(items) >= max_items:
            break
    return items


def compare_sources(docs, query: str, max_results=12):
    query_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", query.lower())
    if not query_tokens:
        return []
    comparisons = []
    for doc in docs:
        text = doc.page_content or ""
        lowered = text.lower()
        overlap = sum(1 for token in query_tokens if token in lowered)
        if overlap == 0:
            continue
        first_hit = min((lowered.find(token) for token in query_tokens if token in lowered), default=0)
        start = max(0, first_hit - 160)
        end = min(len(text), first_hit + 260)
        excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
        comparisons.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "excerpt": excerpt,
            "score": overlap,
        })
    comparisons.sort(key=lambda item: (-item["score"], item["source"], str(item["page"])))
    deduped = []
    seen = set()
    for item in comparisons:
        key = (item["source"], str(item["page"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break
    return deduped


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
You are a critical editor. Review the following summary for potential bias, missing counter-evidence, or over-reliance on a single source.
List any issues you find, and suggest what a balanced summary would include.

Summary:
{summary}

Critique:
"""
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        return f"Bias check failed: {exc}"


def translate_text(text, target_lang, llm, source_lang):
    prompt = f"Översätt följande text till {target_lang}:\n\n{text}" if source_lang == "sv" else f"Translate the following text to {target_lang}:\n\n{text}"
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        return f"Translation failed: {exc}"


def extract_keywords(text, top_n=10, language="en"):
    if yake is None:
        words = re.findall(r"[A-Za-zÅÄÖåäö0-9]{4,}", text.lower())
        seen = []
        for word in words:
            if word not in seen:
                seen.append(word)
            if len(seen) >= top_n:
                break
        return seen
    kw_extractor = yake.KeywordExtractor(lan=language, top=top_n)
    return [kw for kw, _score in kw_extractor.extract_keywords(text)]


def analyze_sentiment(text, pipe):
    if not pipe:
        return {"label": "N/A", "score": 0.0}
    try:
        return pipe(text[:512])[0]
    except Exception:
        return {"label": "Error", "score": 0.0}
