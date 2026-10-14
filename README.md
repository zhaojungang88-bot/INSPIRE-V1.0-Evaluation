# INSPIRE V1.0

Basic public code for the INSPIRE V1.0 retrieval pipeline associated with JMIR Medical Informatics manuscript #108320.

This repository contains the core control flow: the case text is the retrieval query, there is no intent-recognition step, documents are split into 512-token chunks with a 64-token overlap, 20 candidates are selected, those candidates are reranked, and whole chunks are packed in rank order up to 10 chunks and 4000 tokens.

## Public in this release

- `src/inspire/pipeline.py`: query construction, chunking, candidate selection, reranking, and evidence packing.
- `config/pipeline.json`: the reported component names and limits.
- `examples/`: one fictional case and two fictional snippets, used only to run the code.

The default run uses a local lexical stand-in so the control flow can be tested without external services. Qwen3-Embedding-8B, Chroma, BAAI/bge-reranker-v2-m3, and DeepSeek-V3-0324 are the named study components. Their weights and the vector index are not included.

## Not in this release

The knowledge base, benchmark case inputs, retained diagnosis lists, scoring records, clinician responses, interview records, and historical run logs are not in this repository. Additional shareable study materials are planned for release after publication, limited to materials that were retained and that copyright, consent, and licensing allow to be shared.

Running this code does not reproduce the benchmark results. The 30%, 50%, and 20% source priorities in the configuration are natural-language instructions, not mixture weights.

## Run

From the repository root:

```bash
python3 -m unittest tests.test_pipeline
python3 -m src.inspire
```

The second command prints the evidence block for the fictional example. It does not return a diagnosis list.

## Evidence limit

Ten chunks of 512 tokens contain 5120 tokens. This code therefore keeps the first 7 whole chunks, 3584 tokens, and stops before the next chunk. The 64-token overlap is stored between neighboring chunks of one document. It is not subtracted while packing. Shorter chunks can fill more than 7 positions, up to the cap of 10, when their sum stays within 4000 tokens.

## License

The code in this repository is released under the MIT License. The license does not cover third-party models, the knowledge base, case reports, or study data.
