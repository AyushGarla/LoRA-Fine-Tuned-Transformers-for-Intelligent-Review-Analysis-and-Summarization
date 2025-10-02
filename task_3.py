# task3_generate.py
# =========================================================
# TASK 3: REVIEW SUMMARIZATION USING GENERATIVE AI (Local)
# - Loads Task 2 artifacts from SAVE_DIR
# - Builds structured "evidence bundles" per meta-category
# - Uses a pretrained generative model (BART/T5) to write blog-style articles
# - Saves per-category evidence (.json) and article (.md) + an index
# =========================================================

import os
import json
import re
import math
import string
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import minmax_scale

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------ #
#            CONFIG              #
# ------------------------------ #

# Edit this if needed to your local artifacts folder:
SAVE_DIR = r"F:\MS\Ironhack\week14\Project_customer_review\LLM_Customer_review_project"

ARTICLES_DIR = os.path.join(SAVE_DIR, "task3_summaries")
REVIEWS_PARQUET = os.path.join(SAVE_DIR, "reviews_with_sentiment_and_category.parquet")
PRODUCTS_PARQUET = os.path.join(SAVE_DIR, "products_with_clusters.parquet")

# Choose a summarization model: BART (good quality) or T5 (smaller/faster)
GEN_MODEL_NAME = "facebook/bart-large-cnn"   # or "t5-small"

os.makedirs(ARTICLES_DIR, exist_ok=True)


# ------------------------------ #
#        MODEL LOADING           #
# ------------------------------ #

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device).eval()


# ------------------------------ #
#     TEXT UTILS / KEYPHRASES    #
# ------------------------------ #

PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})
STOP = set(ENGLISH_STOP_WORDS) | {
    "amazon","product","item","buy","bought","purchase","purchased","use","used","using",
    "get","got","make","made","one","also","would","could","still","really","review","reviews"
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().translate(PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ngram_phrases(texts, ngram_range=(1,3), top_k=20, extra_stop=()):
    stop = STOP | set(extra_stop)
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=2,
        stop_words=stop,
        strip_accents="unicode",
        lowercase=True
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    terms = np.array(vec.get_feature_names_out())
    scores = np.asarray(X.sum(axis=0)).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [terms[i] for i in top_idx]


# ------------------------------ #
#        DATA LOADING            #
# ------------------------------ #

def load_task2_frames():
    if not os.path.exists(REVIEWS_PARQUET):
        raise FileNotFoundError(f"Missing: {REVIEWS_PARQUET}")
    if not os.path.exists(PRODUCTS_PARQUET):
        raise FileNotFoundError(f"Missing: {PRODUCTS_PARQUET}")

    reviews = pd.read_parquet(REVIEWS_PARQUET)
    products = pd.read_parquet(PRODUCTS_PARQUET)

    reviews["reviews"] = reviews["reviews"].astype(str).apply(normalize_text)

    if "sentiment_pred" not in reviews.columns:
        def rate_to_sent(r):
            try:
                r = float(r)
            except:
                return "neutral"
            if r <= 2: return "negative"
            if r == 3: return "neutral"
            return "positive"
        reviews["sentiment_pred"] = reviews.get("reviews.rating", np.nan).apply(rate_to_sent)

    return reviews, products


# ------------------------------ #
#   PRODUCT STATS / RANKING      #
# ------------------------------ #

def product_stats_in_category(reviews_cat: pd.DataFrame):
    if "reviews.rating" in reviews_cat.columns:
        avg_star = reviews_cat.groupby("product_id")["reviews.rating"].mean().rename("avg_star")
    else:
        star_map = {"positive": 4.5, "neutral": 3.0, "negative": 1.5}
        avg_star = reviews_cat.groupby("product_id")["sentiment_pred"].apply(
            lambda s: np.mean([star_map.get(x, 3.0) for x in s])
        ).rename("avg_star")

    size = reviews_cat.groupby("product_id")["reviews"].size().rename("num_reviews")

    val_counts = reviews_cat.groupby(["product_id","sentiment_pred"])["reviews"].size().unstack(fill_value=0)
    for c in ["positive","neutral","negative"]:
        if c not in val_counts.columns: val_counts[c] = 0
    total = val_counts.sum(axis=1).replace(0, 1)
    pos_ratio = (val_counts["positive"] / total).rename("pos_ratio")
    neg_ratio = (val_counts["negative"] / total).rename("neg_ratio")

    names = reviews_cat.groupby("product_id")["product_name"].agg(
        lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""
    ).rename("product_name")

    dfp = pd.concat([names, size, avg_star, pos_ratio, neg_ratio], axis=1).reset_index()
    return dfp

def score_products(dfp: pd.DataFrame):
    if len(dfp) == 0:
        return dfp.assign(score=[])
    df = dfp.copy()
    df["avg_star_norm"] = minmax_scale(df["avg_star"])
    df["log_reviews"] = np.log1p(df["num_reviews"])
    if df["log_reviews"].max() > 0:
        df["log_reviews"] = df["log_reviews"] / df["log_reviews"].max()
    df["score"] = 0.5*df["avg_star_norm"] + 0.3*df["pos_ratio"] + 0.2*df["log_reviews"]
    return df


# ------------------------------ #
#     PROS/CONS PER PRODUCT      #
# ------------------------------ #

def product_keyphrases(reviews_cat: pd.DataFrame, product_id: str, top_k=6, brand_words=()):
    sub = reviews_cat[reviews_cat["product_id"] == product_id]
    pos_texts = sub.loc[sub["sentiment_pred"]=="positive","reviews"].tolist()
    neg_texts = sub.loc[sub["sentiment_pred"]=="negative","reviews"].tolist()
    pros = ngram_phrases(pos_texts, top_k=top_k, extra_stop=brand_words)[:3]
    cons = ngram_phrases(neg_texts, top_k=top_k, extra_stop=brand_words)[:3]
    return pros, cons

def differences_among(top3_info):
    diffs = []
    pros_lists = [set(p.get("pros", [])) for p in top3_info]
    for i, p in enumerate(top3_info):
        others = set().union(*[pros_lists[j] for j in range(len(pros_lists)) if j!=i])
        unique = list(set(p.get("pros", [])) - others)
        if unique:
            diffs.append(f"{p['name']}: {', '.join(unique[:2])}")
    return diffs[:4] if diffs else []


# ------------------------------ #
#       EVIDENCE BUILDING        #
# ------------------------------ #

def build_evidence_for_category(cat_name: str, reviews_all: pd.DataFrame):
    reviews_cat = reviews_all[reviews_all["cluster_label"] == cat_name].copy()
    if len(reviews_cat) < 10:
        return None

    brand_words = set()
    for nm in reviews_cat["product_name"].dropna().astype(str).tolist():
        for w in normalize_text(nm).split():
            if len(w) > 2: brand_words.add(w)

    stats = product_stats_in_category(reviews_cat)
    ranked = score_products(stats).sort_values("score", ascending=False)

    top3 = ranked.head(3)
    worst = ranked[ranked["num_reviews"] >= max(3, int(ranked["num_reviews"].median()/2) or 1)].tail(1)
    worst = worst.iloc[0] if len(worst) else ranked.tail(1).iloc[0]

    top3_info = []
    for _, row in top3.iterrows():
        pid = row["product_id"]
        pros, cons = product_keyphrases(reviews_cat, pid, top_k=8, brand_words=brand_words)
        top3_info.append({
            "id": pid,
            "name": row["product_name"],
            "avg_star": float(row["avg_star"]),
            "num_reviews": int(row["num_reviews"]),
            "pos_ratio": float(row["pos_ratio"]),
            "neg_ratio": float(row["neg_ratio"]),
            "pros": pros,
            "complaints": cons
        })

    wpid = worst["product_id"]
    _, w_cons = product_keyphrases(reviews_cat, wpid, top_k=8, brand_words=brand_words)
    worst_info = {
        "id": wpid,
        "name": worst["product_name"],
        "avg_star": float(worst["avg_star"]),
        "num_reviews": int(worst["num_reviews"]),
        "pos_ratio": float(worst["pos_ratio"]),
        "neg_ratio": float(worst["neg_ratio"]),
        "top_complaints": w_cons[:3]
    }

    diffs = differences_among(top3_info)

    evidence = {
        "category": cat_name,
        "top_products": top3_info,
        "differences": diffs,
        "worst_product": worst_info,
        "method_note": "Ranking = 0.5*normalized average star + 0.3*positive ratio + 0.2*log(review_count). Keyphrases via TF-IDF on positive/negative subsets."
    }
    return evidence


# ------------------------------ #
#         GENERATION API         #
# ------------------------------ #

def evidence_to_prompt(e):
    lines = []
    lines.append(f"Category: {e['category']}")
    lines.append("Write a concise, factual, blog-style recommendation article (~350-500 words).")
    lines.append("Sections required: Top 3 products (with key differences), Top complaints per product, Worst product and why to avoid. Use only the facts provided. Do not invent specs.")
    lines.append("")
    lines.append("Top 3 products:")
    for i, p in enumerate(e["top_products"], 1):
        lines.append(f"{i}. {p['name']} (avg_star={p['avg_star']:.2f}, reviews={p['num_reviews']}, pos_ratio={p['pos_ratio']:.2f})")
        if p["pros"]:
            lines.append(f"   Pros: {', '.join(p['pros'][:3])}")
        if p["complaints"]:
            lines.append(f"   Complaints: {', '.join(p['complaints'][:3])}")
    if e.get("differences"):
        lines.append("")
        lines.append("Key differences among top picks:")
        for d in e["differences"]:
            lines.append(f"- {d}")
    lines.append("")
    w = e["worst_product"]
    lines.append(f"Worst product: {w['name']} (avg_star={w['avg_star']:.2f}, reviews={w['num_reviews']})")
    if w.get("top_complaints"):
        lines.append(f"Top complaints: {', '.join(w['top_complaints'][:3])}")
    lines.append("")
    lines.append(f"Method note: {e['method_note']}")
    return "\n".join(lines)

def generate_article(prompt: str, max_new_tokens=450):
    inputs = gen_tokenizer([prompt], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# ------------------------------ #
#            RUNNER              #
# ------------------------------ #

def run_task3():
    print("Loading Task 2 frames...")
    reviews, products = load_task2_frames()

    categories = sorted(reviews["cluster_label"].dropna().unique().tolist())
    if not categories:
        raise RuntimeError("No categories found in reviews. Ensure Task 2 artifacts are correct.")

    index_entries = []
    for cat in categories:
        print(f"\nProcessing category: {cat}")
        evidence = build_evidence_for_category(cat, reviews)
        if evidence is None:
            print(f"  Skip '{cat}' (not enough data).")
            continue

        prompt = evidence_to_prompt(evidence)
        article = generate_article(prompt, max_new_tokens=450)

        base = re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")
        ev_path = os.path.join(ARTICLES_DIR, f"{base}_evidence.json")
        md_path = os.path.join(ARTICLES_DIR, f"{base}_article.md")

        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Best {cat}: Top Picks and One to Skip\n\n")
            f.write(article.strip() + "\n")

        print(f"  Saved: {md_path}")
        index_entries.append({
            "category": cat,
            "evidence_path": ev_path,
            "article_path": md_path
        })

    idx_path = os.path.join(ARTICLES_DIR, "category_summaries_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)

    print(f"\nIndex saved: {idx_path}")
    return idx_path


if __name__ == "__main__":
    run_task3()
