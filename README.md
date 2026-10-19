# Relevance filter for agent web context

Question: can a class-balanced LightGBM relevance filter keep more relevant pages in an agent's top-5 context than a TF-IDF top-5 retrieval baseline?

Setup:
- Stand-in pages: 20 Newsgroups posts (headers, footers and quotes removed), truncated to 1,500 characters.
- Categories are split once (seed 42) into 13 training categories and 7 unseen test categories. Each category name is a query.
- Baseline: TF-IDF cosine retrieval, top 5 pages per query (unchanged).
- Improved: class-balanced LightGBM over query-page pairs. Features are TF-IDF cosine, cosine relative to the query's best score, within-query z-score and rank of the cosine, query-term coverage, a 100-dimensional LSA cosine (with its within-query z-score), and log page length. Hyper-parameters are chosen by 4-fold CV over training queries only.
- Metric: precision@5 on test queries, averaged over three LightGBM seeds. Success@5 and tokens read per query are reported as secondary metrics.

Run: `python analysis.py` (writes `results.json` and `figures/`). Requires numpy, scikit-learn, lightgbm, matplotlib.

Caveat: there are only seven test queries, so the results are high-variance.

## Results (latest run)

On 20 Newsgroups pages as a web-page stand-in (7 unseen test queries), a standard TF-IDF top-5 retrieval baseline reached 69% precision@5 while my class-balanced LightGBM filter reached 70%, a small, high-variance gain that read more tokens per query (1,088 vs 758), not fewer.

| Metric | Value |
|---|---|
| baseline_score | 0.6857 |
| improved_score | 0.7048 |
| improvement | 0.01905 |
| baseline_score_std | 1.11e-16 |
| improved_score_std | 0.02694 |
| baseline_success_at_5 | 1 |
| improved_success_at_5 | 1 |
| baseline_tokens_per_query | 758.2 |
| improved_tokens_per_query | 1088 |
| tokens_saving_per_query | -329.7 |
| cv_precision_best | 0.6083 |
| chosen_num_leaves | 15 |
| chosen_n_estimators | 100 |
| n_test_queries | 7 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in corpus, not the paper's web-agent setup (no LLM agent, no real web pages). Only seven test queries, so results are high-variance; the reported std only reflects LightGBM seed variation, not the choice of category split. Test categories are 7 of 20 (35% of categories, a deliberate category-level split), which is above the 20-30% guideline. Vectorizers and LSA are fitted unsupervised on each pool's pages; the TF-IDF baseline is unchanged. Hyper-parameters were chosen by query-level CV on training categories only; the test pool was never used for selection. fetch_20newsgroups has no timeout option, so a 60 s socket timeout and retry/backoff wrap the download.

## Reference

This project is motivated by the following paper:

> A Steiner, R Peeters, C Bizer (2026). *MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web*. Proceedings of the ACM Web Conference. https://dl.acm.org/doi/abs/10.1145/3774904.3792893

Paper author: Prof. Dr. Christian Bizer, Universität Mannheim (christian.bizer@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
