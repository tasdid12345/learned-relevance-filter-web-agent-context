"""Relevance filter for agent web context: TF-IDF top-5 baseline vs class-balanced LightGBM.

Stand-in web pages = 20 Newsgroups posts truncated to 1,500 characters. Categories are split
into training and unseen test categories; each category name is a search query.
"""
import json, os, socket, sys, time
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

SEED, SEEDS = 42, [42, 43, 44]
K = 5                     # pages placed in the agent's context
PAGE_CHARS = 1500         # each post is truncated to this many characters
PAGES_PER_CAT = 150
N_TEST_CATS = 7           # remaining categories form the training pool
GRID = [(7, 100), (7, 300), (15, 100), (15, 300)]  # (num_leaves, n_estimators)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fetch_with_retry(max_tries=5):
    """fetch_20newsgroups has no timeout/retry options: use a 60 s socket timeout and
    retry HTTP 429/5xx and network errors with exponential backoff."""
    socket.setdefaulttimeout(60)
    for attempt in range(max_tries):
        try:
            return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        except Exception as e:
            code = getattr(e, "code", None)  # HTTP status; None for network errors
            retryable = code is None or code == 429 or code >= 500
            if attempt == max_tries - 1 or not retryable:
                raise
            wait = 5 * 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def sample_pages(data, cat_ids, rng):
    """Sample up to PAGES_PER_CAT non-empty posts per category, truncated to PAGE_CHARS."""
    texts, labels = [], []
    for c in cat_ids:
        idx = [i for i in np.where(data.target == c)[0] if data.data[i].strip()]
        for i in rng.choice(idx, size=min(PAGES_PER_CAT, len(idx)), replace=False):
            texts.append(data.data[i][:PAGE_CHARS])
            labels.append(c)
    return texts, np.array(labels)


def pair_features(vec, queries, pages):
    """Cosine matrix [Q,P] and per-pair features [Q,P,4]: cosine, cosine relative to the
    query's best score, query-term coverage, log page length."""
    cos = (vec.transform(queries) @ vec.transform(pages).T).toarray()  # rows are L2-normalised
    rel_cos = cos / (cos.max(axis=1, keepdims=True) + 1e-12)
    analyze = vec.build_analyzer()
    page_terms = [set(analyze(p)) for p in pages]
    q_terms = [set(analyze(q)) for q in queries]
    cov = np.array([[len(qt & pt) / max(len(qt), 1) for pt in page_terms] for qt in q_terms])
    loglen = np.broadcast_to(np.log1p([len(p) for p in pages]), cos.shape)
    return cos, np.stack([cos, rel_cos, cov, loglen], axis=-1)


def evaluate(scores, rel, page_len):
    """Precision@K, success@K and mean tokens read (chars/4) of the top-K pages per query."""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    hits = np.take_along_axis(rel, top, axis=1)
    return {"precision": float(hits.mean()),
            "success": float((hits.sum(1) > 0).mean()),
            "tokens": float((page_len[top].sum(1) / 4).mean())}


def make_model(leaves, n_est, seed):
    # class_weight="balanced" compensates for ~1 relevant page per 13 pairs
    return lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=0.05,
                              class_weight="balanced", subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)


def fit_predict(F_fit, R_fit, F_pred, leaves, n_est, seed):
    """Train on query-page pairs; return predicted relevance probabilities [Q,P]."""
    m = make_model(leaves, n_est, seed).fit(F_fit.reshape(-1, F_fit.shape[-1]), R_fit.reshape(-1))
    return m.predict_proba(F_pred.reshape(-1, F_pred.shape[-1]))[:, 1].reshape(F_pred.shape[:2])


def main():
    data = fetch_with_retry()
    log(f"loaded {len(data.data)} posts, {len(data.target_names)} categories")

    # One fixed category-level split (seed 42): training pool vs unseen test pool
    perm = np.random.RandomState(SEED).permutation(len(data.target_names))
    test_cats, train_cats = perm[:N_TEST_CATS], perm[N_TEST_CATS:]
    rng = np.random.RandomState(SEED)
    tr_pages, tr_lab = sample_pages(data, train_cats, rng)
    te_pages, te_lab = sample_pages(data, test_cats, rng)
    names = lambda ids: [data.target_names[c].replace(".", " ") for c in ids]
    tr_q, te_q = names(train_cats), names(test_cats)
    R_tr = (tr_lab[None, :] == train_cats[:, None]).astype(int)  # relevance [Q,P]
    R_te = (te_lab[None, :] == test_cats[:, None]).astype(int)
    len_tr = np.array([len(p) for p in tr_pages])
    len_te = np.array([len(p) for p in te_pages])
    log(f"train pool: {len(tr_pages)} pages/{len(tr_q)} queries; test pool: {len(te_pages)}/{len(te_q)}")

    # Unsupervised TF-IDF fitted on each pool's own pages; identical for both models
    vec_tr = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(tr_pages)
    vec_te = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(te_pages)
    _, F_tr = pair_features(vec_tr, tr_q, tr_pages)
    cos_te, F_te = pair_features(vec_te, te_q, te_pages)

    # Hyper-parameter selection: 4-fold CV over TRAINING queries only (metric: precision@5)
    folds = list(KFold(4, shuffle=True, random_state=SEED).split(np.arange(len(tr_q))))
    cv = {}
    for cfg in GRID:
        scores = []
        for fit_i, val_i in folds:
            p = fit_predict(F_tr[fit_i], R_tr[fit_i], F_tr[val_i], *cfg, SEED)
            scores.append(evaluate(p, R_tr[val_i], len_tr)["precision"])
        cv[cfg] = float(np.mean(scores))
        log(f"CV leaves={cfg[0]} n_est={cfg[1]}: precision@5={cv[cfg]:.4f}")
    best = max(GRID, key=lambda c: cv[c])
    log(f"selected leaves={best[0]} n_estimators={best[1]}")

    # Final evaluation on the unseen test pool, once per seed
    runs = {"baseline": [], "improved": []}
    for seed in SEEDS:  # baseline is deterministic; repeated so both follow the same protocol
        runs["baseline"].append(evaluate(cos_te, R_te, len_te))
        runs["improved"].append(evaluate(fit_predict(F_tr, R_tr, F_te, *best, seed), R_te, len_te))
        log(f"seed {seed}: P@5 baseline={runs['baseline'][-1]['precision']:.4f} "
            f"improved={runs['improved'][-1]['precision']:.4f}")

    def stat(model, key):
        v = np.array([r[key] for r in runs[model]])
        return float(v.mean()), float(v.std())

    bp, bp_sd = stat("baseline", "precision")
    ip, ip_sd = stat("improved", "precision")
    if not np.isfinite([bp, ip]).all():
        log("non-finite metrics; cannot produce meaningful results")
        sys.exit(1)
    bs, _ = stat("baseline", "success")
    is_, _ = stat("improved", "success")
    bt, _ = stat("baseline", "tokens")
    it, _ = stat("improved", "tokens")

    # Charts
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["TF-IDF top-5", "LightGBM top-5"], [bp, ip], yerr=[bp_sd, ip_sd],
           color=["gray", "tab:blue"])
    ax.set_ylabel("precision@5 (test)")
    fig.tight_layout(); fig.savefig("figures/precision_at_5.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["TF-IDF top-5", "LightGBM top-5"], [bt, it], color=["gray", "tab:blue"])
    ax.set_ylabel("mean tokens read per query (chars/4)")
    fig.tight_layout(); fig.savefig("figures/tokens_per_query.png", dpi=120); plt.close(fig)

    results = {
        "question": "Can a class-balanced LightGBM relevance filter over web pages keep more relevant "
                    "pages in an agent's top-5 context while reading fewer characters than a standard "
                    "TF-IDF retrieval baseline?",
        "method": "Posts from 20 Newsgroups (headers/footers/quotes removed) are pages truncated to "
                  "1,500 characters; categories are split into training and unseen test categories; "
                  "category names are queries. Both models rank the test pages per query; metric is "
                  "precision@5 with success@5 and tokens read as secondary metrics.",
        "dataset": {
            "name": "20 Newsgroups (scikit-learn fetch_20newsgroups) as web-page stand-in",
            "source_url": "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset",
            "n_rows": len(tr_pages) + len(te_pages),
            "n_train": len(tr_pages),
            "n_test": len(te_pages),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "precision@5 (test queries)",
        "higher_is_better": True,
        "baseline": "TF-IDF cosine similarity retrieval, top-5 pages (RAG-style retriever).",
        "improved": "Class-balanced LightGBM classifier trained on query-page pairs from training "
                    "categories, with TF-IDF cosine, cosine relative to the query's best score, "
                    "query-term coverage and log page length; ranks test pages by predicted probability, top 5.",
        "metrics": {
            "baseline_score": bp,
            "improved_score": ip,
            "improvement": ip - bp,
            "baseline_score_std": bp_sd,
            "improved_score_std": ip_sd,
            "baseline_success_at_5": bs,
            "improved_success_at_5": is_,
            "baseline_tokens_per_query": bt,
            "improved_tokens_per_query": it,
            "tokens_saving_per_query": bt - it,
            "cv_precision_best": cv[best],
            "chosen_num_leaves": best[0],
            "chosen_n_estimators": best[1],
            "n_test_queries": len(te_q),
        },
        "notes": "Stand-in corpus, not the paper's web-agent setup (no LLM agent, no real web pages). "
                 "Only seven test queries, so results are high-variance; the reported std only reflects "
                 "LightGBM seed variation, not the choice of category split. Test categories are 7 of 20 "
                 "(35% of categories, a deliberate category-level split), which is above the 20-30% "
                 "guideline. Vectorizers are fitted unsupervised on each pool's pages. Hyper-parameters "
                 "were chosen by query-level CV on training categories only. fetch_20newsgroups has no "
                 "timeout option, so a 60 s socket timeout and retry/backoff wrap the download.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"done: P@5 baseline={bp:.4f} improved={ip:.4f}; results.json written")


if __name__ == "__main__":
    main()
