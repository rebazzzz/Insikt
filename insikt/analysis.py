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
