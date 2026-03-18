# Insikt Early Testing Guide

Use this guide when sharing Insikt with a small group of early testers.

## Who To Invite

Start with 3 to 5 people who match the real target audience:

- Journalists or researchers who work with PDFs, transcripts, and notes
- At least one non-technical tester
- At least one heavy-document tester with large or scanned files
- At least one bilingual Swedish/English tester if that workflow matters

## What To Send Them

Send these items together:

- The app folder or repo
- For non-technical Windows testers: `INSTALL_FOR_TESTER.bat` and `START_INSIKT.bat`
- A short install/start message
- A few suggested tasks to try
- The feedback template in [FEEDBACK_TEMPLATE.md](/c:/Users/Changers%20Hub/Desktop/INSIKT/FEEDBACK_TEMPLATE.md)

## Recommended Start Message

Use something close to this:

```text
Please test Insikt as if you were doing real research work.

What to try:
1. Load one or more documents
2. Ask questions in chat
3. Generate a summary
4. Try quote extraction, timeline, source comparison, and writing draft tools
5. Export something

Please note:
- What felt confusing
- Where the app felt slow
- Anything you did not trust
- Any bugs, crashes, or wrong answers

Please send feedback using the attached template.
```

## Core Scenarios To Ask Testers To Try

1. Upload a normal text PDF and ask factual questions with citations.
2. Upload a scanned PDF and confirm OCR text can be used in chat or summaries.
3. Generate a summary and check whether the confidence label feels believable.
4. Use quote extraction and verify whether quotes and page references are useful.
5. Use source comparison across two documents with conflicting details.
6. Generate a writing draft and review the claim checker output.
7. Save a case, reload it, and confirm the session browser makes sense.
8. Export a result and confirm the source appendix/endnotes are understandable.

## What You Should Watch For

- First-run confusion
- Missing dependencies on tester machines
- Slow ingest on large files
- Weak OCR on poor scans
- Citation trust problems
- Export formatting issues
- Session save/load confusion
- Swedish/English wording that feels awkward

## Known Limits To Tell Testers Up Front

- The app is local-first and depends on the tester's own machine performance
- OCR quality depends heavily on scan quality
- LLM output still needs editorial verification
- Ollama models must be available locally for chat and generation features

## Minimum Bar Before A Wider Beta

- Testers can start the app without help
- Text PDFs and scanned PDFs both work
- Chat, summary, writing, and export work on real source material
- Error messages are understandable enough for non-technical users
- Feedback shows more workflow value than confusion
