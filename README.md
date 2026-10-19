# Class-Balanced LightGBM Relevance Filter vs TF-IDF Retrieval for Agent Web Context

Agents that read the web through retrieval (RAG-style interfaces) must choose which pages enter a limited context window. The paper "MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web" (2026) compares how effectively and efficiently different interfaces deliver web content to agents, so the cost of each page in tokens matters as much as whether relevant pages are found. This experiment asks whether a learned relevance filter can improve on a plain retriever on both counts.

**Question:** Can a class-balanced LightGBM relevance filter over web pages keep more relevant pages in an agent's top-5 context while reading fewer characters than a standard TF-IDF retrieval baseline?

**Dataset:** The public 20 Newsgroups corpus (via scikit-learn's `fetch_20newsgroups`, headers, footers and quotes removed) stands in for web pages. Each post is truncated to its first 1,500 characters. Categories are split into training and unseen test categories, and each category name is used as a query. This is not the paper's own dataset.

**Baseline:** TF-IDF cosine similarity retrieval, returning the top five pages per query.

**Improved model:** A class-balanced LightGBM classifier trained on query-page pairs from the training categories. Its features are TF-IDF cosine, cosine relative to the query's best score, query-term coverage and log page length. Test pages are ranked by predicted probability, and the top five are kept.

**Metrics:** Main metric is precision@5. Secondary metrics are success@5 and mean tokens read per query (characters/4).

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts saved to `figures/`).

## Reference

This project is motivated by the following paper:

> A Steiner, R Peeters, C Bizer (2026). *MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web*. Proceedings of the ACM Web Conference. https://dl.acm.org/doi/abs/10.1145/3774904.3792893

Paper author: Prof. Dr. Christian Bizer, Universität Mannheim (christian.bizer@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
