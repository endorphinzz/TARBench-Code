# Token Cost Analysis (estimated, tiktoken cl100k_base)

Per-sample tokens amortized over 5-sample batched calls.

| Condition | Model | Input tok/sample | Output tok/sample | Mean input tok/call | Mean output tok/call |
|---|---|---:|---:|---:|---:|
| zero-shot | gpt-5.4 | 346 | 59 | 1732 | 295 |
| zero-shot | deepseek-v4-flash | 346 | 46 | 1732 | 228 |
| zero-shot | gemini-3.1-flash-lite-preview | 346 | 52 | 1732 | 261 |
| ICL-1 | gpt-5.4 | 2179 | 54 | 10896 | 272 |
| ICL-1 | deepseek-v4-flash | 2179 | 48 | 10896 | 242 |
| ICL-1 | gemini-3.1-flash-lite-preview | 2179 | 48 | 10896 | 237 |
| ICL-3 | gpt-5.4 | 3029 | 55 | 15146 | 274 |
| ICL-3 | deepseek-v4-flash | 3029 | 49 | 15146 | 244 |
| ICL-3 | gemini-3.1-flash-lite-preview | 3029 | 51 | 15146 | 256 |
| ICL-5 | gpt-5.4 | 3877 | 56 | 19386 | 278 |
| ICL-5 | deepseek-v4-flash | 3877 | 50 | 19386 | 251 |
| ICL-5 | gemini-3.1-flash-lite-preview | 3877 | 52 | 19386 | 260 |
| no-retrieval | gpt-5.4 | 363 | 58 | 1813 | 292 |
| no-retrieval | deepseek-v4-flash | 363 | 46 | 1813 | 233 |
| no-retrieval | gemini-3.1-flash-lite-preview | 363 | 51 | 1813 | 255 |
| no-retrieval | claude-sonnet-4-5 | 363 | 55 | 1813 | 276 |
| RAG | gpt-5.4 | 6318 | 54 | 31590 | 272 |
| RAG | deepseek-v4-flash | 6318 | 78 | 31590 | 392 |
| RAG | gemini-3.1-flash-lite-preview | 6318 | 50 | 31590 | 251 |
| RAG | claude-sonnet-4-5 | 6318 | 55 | 31590 | 277 |
