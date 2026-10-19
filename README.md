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

## Reference

This project is motivated by the following paper:

> A Steiner, R Peeters, C Bizer (2026). *MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web*. Proceedings of the ACM Web Conference. https://dl.acm.org/doi/abs/10.1145/3774904.3792893

Paper author: Prof. Dr. Christian Bizer, Universität Mannheim (christian.bizer@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
