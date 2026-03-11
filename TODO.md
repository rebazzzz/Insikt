# TODO: Performance Optimizations for Insikt App

## Phase 1: Quantized Models ✅ COMPLETED

- [x] Added LLM_MODELS configuration with multiple model options
- [x] Added quantized models: llama3.2:3b (fast), llama3.2:1b (turbo)
- [x] Default set to llama3.2:3b for optimal speed/quality balance
- [x] User-friendly descriptions with emoji indicators (⚡ speed, ⭐ quality)
- [x] Updated load_llm to use selected model from session state

## Phase 2: Session Caching ✅ COMPLETED

- [x] Document hash-based caching already implemented
- [x] Summary caching with hash keys
- [x] Session persistence to disk (FAISS vectorstore + pickle)
- [x] Cached summaries retrieved automatically

## Phase 3: Semantic Chunking ✅ COMPLETED

- [x] Implemented semantic_chunking function using embeddings
- [x] Uses cosine similarity to group semantically related sentences
- [x] Default strategy set to "semantic" for better accuracy
- [x] Fallback to fixed-size chunking if semantic fails
- [x] Merges small chunks and splits large ones for optimal size

## Additional Updates:

- [x] Added numpy to requirements.txt for similarity calculations
- [x] Added llm_model and chunking_strategy to session state defaults
- [x] All optimizations work automatically with smart defaults
- [x] Non-tech friendly descriptions throughout

## Implementation Summary:

### 7. Quantized Models:

- Default: llama3.2:3b (fast, good quality, medium memory)
- Options: llama3.2:1b (turbo, fastest, lowest memory), llama3.2 (standard, best quality)
- All models pre-configured with user-friendly descriptions

### 8. Session Caching:

- Hash-based document caching for summaries
- FAISS vectorstore persisted to disk
- Session data saved/loaded with all settings

### 9. Semantic Chunking:

- Uses embeddings to find semantically similar sentences
- Groups sentences based on cosine similarity threshold (0.5)
- Better for maintaining context in documents
- Falls back to fixed-size if semantic fails
