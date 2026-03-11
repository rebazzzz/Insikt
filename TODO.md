# TODO: Enhanced RAG Improvements for Insikt App

## Phase 1: Better Embeddings ✅ COMPLETED

- [x] Upgrade from "all-MiniLM-L6-v2" to "BAAI/bge-base-en-v1.5"
- [x] Add sidebar setting for embedding model selection (small/base)
- [x] Update load_embeddings function to support model selection

## Phase 2: Adaptive Retrieval ✅ COMPLETED

- [x] Create analyze_query_complexity function
- [x] Implement dynamic k selection based on query complexity
- [x] Simple queries: k=3-4
- [x] Medium complexity: k=5-7
- [x] Complex analytical: k=8-10

## Phase 3: Citation Verification ✅ COMPLETED

- [x] Create verify_citations function
- [x] Extract citations from generated response
- [x] Verify cited sources exist in retrieved context
- [x] Add verification step in chat_with_docs
- [x] Option to regenerate if citations are invalid

## Summary of Changes:

### 4. Better Embeddings:

- Upgraded to "BAAI/bge-base-en-v1.5" for better quality
- Added user-selectable model in sidebar (bge-small/bge-base options)
- Better semantic understanding and retrieval accuracy
- Session state preserves user choice

### 5. Adaptive Retrieval:

- Query complexity analysis based on:
  - Query length (word count)
  - Analytical keywords (compare, analyze, explain, why, how)
  - Multiple entities/concepts
  - Comparison indicators
- Dynamic k selection: k=3 (simple), k=5 (medium), k=8 (complex)

### 6. Citation Verification:

- Extracts citations from response (handles both Swedish and English formats)
- Verifies citations exist in retrieved context
- Adds warning if citations cannot be verified
- Returns verified response with any warnings
