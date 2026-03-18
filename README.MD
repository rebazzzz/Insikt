# Insikt

Insikt is a local-first journalist research and writing app for chatting with documents, generating summaries, extracting analysis, and exporting grounded drafts.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Make sure Ollama is installed and running.
4. Start the app with `streamlit run insikt_app.py` or `start.bat`.

For scanned PDFs, install the Tesseract OCR CLI too.

## Non-Technical Tester Install

For a simple Windows tester flow:

1. Run [INSTALL_FOR_TESTER.bat](/c:/Users/Changers%20Hub/Desktop/INSIKT/INSTALL_FOR_TESTER.bat) once.
2. Then launch the app with [START_INSIKT.bat](/c:/Users/Changers%20Hub/Desktop/INSIKT/START_INSIKT.bat).

This path is meant for early testers who should not need to manage Python manually.

## Focus Areas

- Local document chat and summarization with citations
- Writing support for reporting, scripting, and editorial workflows
- Source-aware analysis and export tools

## Roadmap

Planned work is tracked in [TODO.md](/c:/Users/Changers%20Hub/Desktop/INSIKT/TODO.md).

## Early Testing

If you are preparing a small tester rollout, use [EARLY_TESTING.md](/c:/Users/Changers%20Hub/Desktop/INSIKT/EARLY_TESTING.md) and collect responses with [FEEDBACK_TEMPLATE.md](/c:/Users/Changers%20Hub/Desktop/INSIKT/FEEDBACK_TEMPLATE.md).

Current roadmap themes:

- Easier onboarding with automatic environment checks and laptop-based model recommendations
- Stronger source review with inline citations, a document navigator, and confidence labels
- Journalist workflows such as quote extraction, contradiction finding, and draft claim checking
- Better document handling through OCR, performance improvements, and richer session management
