# Insikt Product Roadmap

This file tracks the next high-impact product work for Insikt.

## Current Priorities

- [x] One-click onboarding
  First-run setup wizard that checks Ollama, installed models, and local dependencies, then offers automatic fixes for common issues.
- [x] Auto model recommendations
  Suggest `Fast`, `Balanced`, and `Best quality` based on the actual laptop while still allowing manual model selection.
- [x] OCR and scanned PDF support
  Add robust OCR so scanned source material works in the same workflow as text PDFs.
- [x] Clickable inline citations
  Keep the citation register, but also make citations inside the generated text directly clickable.
- [x] Document navigator
  Add a left-side source panel with filename, page navigation, bookmarks, and highlighted cited passages.
- [x] Better session browser
  Replace simple save names with thumbnails, timestamps, tags, and case folders.
- [x] Accessibility and simplicity
  Introduce larger controls, plainer wording, fewer technical labels, and a `basic mode` for non-technical users.

## Reporter Workflow

- [x] Reporter workflow templates
  Add presets such as `summarize interview`, `build timeline`, `find contradictions`, `extract names and roles`, and `write article draft`.
- [x] Quote extraction
  Pull exact quote candidates with page references so reporters can move faster from source review to drafting.
- [x] Source comparison mode
  Compare what multiple documents say about the same person, event, or claim.
- [x] Claim checker
  Highlight statements in a generated draft that still need verification before publication.
- [x] Case board
  Create a shared workspace for notes, pinned excerpts, important people, dates, and draft angles.

## Trust, Review, And Export

- [x] Answer confidence mode
  Show plain-language evidence signals such as `well-supported`, `partly supported`, or `needs review`.
- [x] Safer exports
  Export with automatic footnotes or endnotes plus a source appendix.

## Performance And Scale

- [x] Large-document performance mode
  Add a background indexing queue, cache visibility, and smarter incremental updates when only one file changes.

## Suggested Delivery Order

### Phase 1: Friction Removal

- [x] One-click onboarding
- [x] Auto model recommendations
- [x] Accessibility and simplicity
- [x] Better session browser

### Phase 2: Source Trust And Navigation

- [x] Clickable inline citations
- [x] Document navigator
- [x] Answer confidence mode
- [x] Safer exports

### Phase 3: Reporter Workflows

- [x] Reporter workflow templates
- [x] Quote extraction
- [x] Source comparison mode
- [x] Claim checker
- [x] Case board

### Phase 4: Heavy-Duty Document Support

- [x] OCR and scanned PDF support
- [x] Large-document performance mode

## Notes

- Inline citation clicks and the document navigator should be designed together so the cited passage opens in context.
- OCR support should plug into the same citation and page-reference model used for native PDFs.
- Confidence labels should be evidence-based and phrased for newsroom use, not as model probability jargon.
