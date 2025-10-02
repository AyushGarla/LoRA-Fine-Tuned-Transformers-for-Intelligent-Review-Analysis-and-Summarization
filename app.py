# app.py
# ============================================
# Marketing Insights Dashboard (Streamlit)
# - Task 1: Live sentiment classification (LoRA transformer)
# - Task 2: Category clustering insights (TF-IDF + KMeans artifacts)
# - Task 3: Category summaries (BART/T5-generated markdown)
# ============================================

import os, json, re, math, string
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting
import altair as alt

# ======= CONFIG (edit these defaults to your paths) =======
# If you're on Colab + Drive, point to your SAVE_DIR from earlier steps.
DEFAULT_ARTIFACTS_DIR = r"F:\MS\Ironhack\week14\Project_customer_review\LLM_Customer_review_project"
DEFAULT_SUMMARIES_DIR = r"F:\MS\Ironhack\week14\Project_customer_review\LLM_Customer_review_project\task3_summaries"
DEFAULT_MODEL_DIR     = r"F:\MS\Ironhack\week14\Project_customer_review\LLM_Customer_review_project"

# ======= CACHING HELPERS =======
@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_dir: str):
    """Load Task-1 LoRA model for live classification."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_CKPT = "nlptown/bert-base-multilingual-uncased-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_CKPT, num_labels=3, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval().to(device)

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    return model, tokenizer, device, id2label

def predict_label(text: str, model, tokenizer, device, max_len=128) -> str:
    import torch
    if not text.strip():
        return ""
    enc = tokenizer([text], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = int(logits.argmax(dim=-1).cpu().item())
    return {0:"negative",1:"neutral",2:"positive"}[pred_id]

# ======= UI LAYOUT =======
st.set_page_config(page_title="Marketing Insights Dashboard", layout="wide")

st.title("📊 Marketing Insights Dashboard")
st.caption("Sentiment Classification • Category Clustering • Generative Summaries")

with st.sidebar:
    st.header("Settings")
    artifacts_dir = st.text_input("Artifacts directory", DEFAULT_ARTIFACTS_DIR)
    summaries_dir = st.text_input("Summaries directory (Task 3)", DEFAULT_SUMMARIES_DIR)
    model_dir = st.text_input("Sentiment model dir (Task 1)", DEFAULT_MODEL_DIR)

    # Expected artifact paths
    products_path = os.path.join(artifacts_dir, "products_with_clusters.parquet")
    reviews_path = os.path.join(artifacts_dir, "reviews_with_sentiment_and_category.parquet")
    meta_path = os.path.join(artifacts_dir, "clustering_metadata.json")
    idx_path = os.path.join(summaries_dir, "category_summaries_index.json")

    st.markdown("**Expected files:**")
    st.code(f"{products_path}\n{reviews_path}\n{meta_path}\n{idx_path}", language="text")

# ======= LOAD DATA SAFELY =======
missing = []
if not os.path.exists(products_path): missing.append(products_path)
if not os.path.exists(reviews_path): missing.append(reviews_path)
if not os.path.exists(meta_path):     missing.append(meta_path)
if not os.path.exists(idx_path):      missing.append(idx_path)

if missing:
    st.error("Missing required files:\n" + "\n".join(missing))
    st.stop()

products_df = load_parquet(products_path)
reviews_df = load_parquet(reviews_path)
meta = load_json(meta_path)
summ_index = load_json(idx_path)

# Ensure required columns exist
if "cluster_label" not in products_df.columns or "cluster_label" not in reviews_df.columns:
    st.error("cluster_label not found. Please run Task 2 pipeline first.")
    st.stop()

# Prepare category list
categories = sorted([c for c in reviews_df["cluster_label"].dropna().unique().tolist() if str(c).strip() != ""])
if not categories:
    st.error("No categories found in reviews. Check Task 2 outputs.")
    st.stop()

# ======= TABS =======
tab_overview, tab_category, tab_classifier = st.tabs([
    "Overview", "Category Insights", "Live Classifier"
])

# ================= OVERVIEW =================
with tab_overview:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Products", f"{products_df['product_id'].nunique():,}")
    with col2:
        st.metric("Total Reviews", f"{len(reviews_df):,}")
    with col3:
        st.metric("Meta-Categories", f"{len(categories)}")

    # Category size chart
    cat_counts = reviews_df["cluster_label"].value_counts().reset_index()
    cat_counts.columns = ["category", "reviews"]
    chart = (
        alt.Chart(cat_counts)
        .mark_bar()
        .encode(x=alt.X("category:N", sort='-y'), y="reviews:Q", tooltip=["category","reviews"])
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Show cluster name suggestions from meta
    st.markdown("### Cluster Name Suggestions (from top terms)")
    cluster_terms = meta.get("cluster_top_terms", {})
    cluster_name_map = meta.get("cluster_name_map", {})
    rows = []
    for cid_str, terms in cluster_terms.items():
        cid = int(cid_str) if isinstance(cid_str, str) and cid_str.isdigit() else cid_str
        rows.append({
            "cluster_id": cid,
            "label": cluster_name_map.get(str(cid), cluster_name_map.get(cid, f"Cluster {cid}")),
            "top_terms": ", ".join(terms[:10])
        })
    if rows:
        st.dataframe(pd.DataFrame(rows))

# ================= CATEGORY INSIGHTS =================
with tab_category:
    st.subheader("Category Insights & Summary")
    sel_cat = st.selectbox("Choose a category", categories, index=0)

    cat_reviews = reviews_df[reviews_df["cluster_label"] == sel_cat].copy()
    if cat_reviews.empty:
        st.warning("No reviews in this category.")
    else:
        # Sentiment distribution (from Task 1 merged or rating fallback)
        if "sentiment_pred" not in cat_reviews.columns:
            # fallback from rating if needed
            def rate_to_sent(r):
                try:
                    r = float(r)
                except:
                    return "neutral"
                if r <= 2: return "negative"
                if r == 3: return "neutral"
                return "positive"
            cat_reviews["sentiment_pred"] = cat_reviews.get("reviews.rating", np.nan).apply(rate_to_sent)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Sentiment distribution**")
            sent_counts = cat_reviews["sentiment_pred"].value_counts().reindex(["negative","neutral","positive"]).fillna(0).astype(int).reset_index()
            sent_counts.columns = ["sentiment", "count"]
            sent_chart = (
                alt.Chart(sent_counts)
                .mark_bar()
                .encode(x=alt.X("sentiment:N", sort=["negative","neutral","positive"]),
                        y="count:Q", tooltip=["sentiment","count"])
                .properties(height=300)
            )
            st.altair_chart(sent_chart, use_container_width=True)

        with colB:
            st.markdown("**Top products by average rating**")
            if "reviews.rating" in cat_reviews.columns:
                top_ratings = (
                    cat_reviews.groupby(["product_id","product_name"])["reviews.rating"]
                    .mean().reset_index().sort_values("reviews.rating", ascending=False).head(10)
                )
                st.dataframe(top_ratings.rename(columns={"reviews.rating":"avg_rating"}))
            else:
                st.info("Numeric star ratings not available in this dataset slice.")

        # ===== Show generated article (Task 3) =====
        st.markdown("### 📝 Category Summary (Generated)")
        # Find article path from index
        def find_article_path(category: str) -> str:
            for row in summ_index:
                if row.get("category") == category:
                    return row.get("article_path", "")
            return ""
        md_path = find_article_path(sel_cat)
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("No generated article found for this category. Run Task 3 to create it.")

# ================= LIVE CLASSIFIER =================
with tab_classifier:
    st.subheader("Live Review Sentiment Classifier (Task 1 Model)")
    st.caption("Type or paste a review below and click **Classify**.")
    user_text = st.text_area("Your review", height=140, placeholder="e.g., I love this e-reader. The battery lasts forever and the screen is crisp.")
    btn = st.button("Classify")
    if btn:
        with st.spinner("Loading model (first time only)..."):
            try:
                model, tokenizer, device, id2label = load_sentiment_model(model_dir)
                label = predict_label(user_text, model, tokenizer, device)
                if label:
                    color = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}[label]
                    st.success(f"Prediction: {color} **{label.capitalize()}**")
                else:
                    st.warning("Please enter some text.")
            except Exception as e:
                st.error(f"Could not load or run the model. Check model_dir.\n\n{e}")

# ======= FOOTER =======
st.markdown("---")
st.caption("Built from Task 1 (Transformers + LoRA), Task 2 (TF-IDF + KMeans), Task 3 (BART/T5 summaries).")
