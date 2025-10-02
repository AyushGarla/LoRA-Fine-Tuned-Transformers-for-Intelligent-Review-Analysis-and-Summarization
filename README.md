LoRA-Fine-Tuned Transformers for Intelligent Review Analysis and Summarization

End-to-end NLP pipeline that classifies, clusters, and summarizes Amazon product reviews, deployed as an interactive Streamlit dashboard.

Task 1 – Sentiment Classification: LoRA-fine-tuned BERT (3 classes: negative / neutral / positive)

Task 2 – Product Category Clustering: TF-IDF + KMeans → 4–6 meta-categories

Task 3 – Summarization: BART (facebook/bart-large-cnn) generates blog-style articles per category

App – Streamlit Dashboard: Browse categories, view insights & generated summaries; live sentiment classifier

Contents

Project Structure

Setup (Local)

Dataset

Task 1 — Review Classification

Task 2 — Product Category Clustering

Task 3 — Review Summarization

Streamlit Dashboard

Troubleshooting

Notes on Reproducibility

Project Structure
repo/
├─ app.py                                  # Streamlit dashboard
├─ task1_classification.py                 # Task 1 training + inference utilities (LoRA + BERT)
├─ task2_clustering.py                     # Task 2 pipeline (TF-IDF + KMeans + mapping)
├─ task3_generate.py                       # Task 3 generator (BART) → markdown articles
├─ requirements.txt
├─ README.md
└─ artifacts/                              # (you create/point this)
   ├─ products_with_clusters.parquet
   ├─ reviews_with_sentiment_and_category.parquet
   ├─ clustering_metadata.json
   ├─ task1_test_predictions.csv/.json     # optional
   ├─ task3_summaries/
   │  ├─ category_summaries_index.json
   │  ├─ <category>_article.md
   │  └─ <category>_evidence.json
   └─ (optional) merged_model/             # merged LoRA weights (see Troubleshooting)
      ├─ config.json
      ├─ pytorch_model.bin / model.safetensors
      └─ tokenizer files


You can name the artifacts directory anything; just point the app to it in the sidebar.

Setup (Local)
1) Create a clean environment

Windows / macOS / Linux (Python 3.10 recommended):

# conda (recommended)
conda create -n reviewsdash python=3.10 -y
conda activate reviewsdash

# or venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


The requirements.txt pins versions that work well together on Windows and avoid DLL issues.

Dataset

This project uses the Kaggle dataset:

datafiniti/consumer-reviews-of-amazon-products (file: 1429_1.csv)

The code loads it via kagglehub (no manual download needed during training if you have internet).
If you prefer, you can manually download the CSV into your project and adjust paths accordingly.

Task 1 — Review Classification

Goal: Train a 3-class sentiment model using LoRA on top of nlptown/bert-base-multilingual-uncased-sentiment.

What it does:

Loads reviews with star ratings from KaggleHub

Maps stars → sentiment: 1–2 = Negative, 3 = Neutral, 4–5 = Positive

Cleans text, splits data, tokenizes, trains with class-weighted loss

Saves:

LoRA adapter + tokenizer (for inference)

Predictions CSV/JSON (optional, for later tasks)

A small test set parquet for demo

Run:

python task1_classification.py


Outputs into your artifacts dir (configure inside the script):

adapter_model.safetensors, adapter_config.json (+ tokenizer files)

task1_test_predictions.csv / .json (optional)

task1_test_df.parquet (held-out sample)

The Streamlit app can load either the LoRA adapter or an optional merged model (see Troubleshooting if adapter fails due to PEFT versions).

Task 2 — Product Category Clustering

Goal: Group products into 4–6 meta-categories using product name+category text.

What it does:

Builds a product table (product_id, product_name, categories_text, product_text)

Vectorizes product_text with TF-IDF (1–2 grams) and L2 normalizes

Tries k ∈ {4,5,6} and picks the best by silhouette score

Extracts top TF-IDF terms per cluster → human-readable cluster labels

Assigns each product and review to a meta-category

Optionally merges Task 1 sentiment predictions if present

Run:

python task2_clustering.py


Outputs into your artifacts dir:

products_with_clusters.parquet

reviews_with_sentiment_and_category.parquet

clustering_metadata.json

category_map.json

Task 3 — Review Summarization

Goal: Generate a short article (blog style) per meta-category with:

Top 3 products + key differences

Top complaints (per product)

Worst product and why to avoid

Model: facebook/bart-large-cnn (or switch to t5-small in the script).

What it does:

Loads Task 2 outputs

Builds evidence bundles per category:

product ranking (normalized stars + positive ratio + review volume)

keyphrases for pros/cons (TF-IDF on pos/neg subsets)

Converts evidence → a concise prompt

Generates Markdown articles and an index JSON

Run:

python task3_generate.py


Outputs into artifacts/task3_summaries/:

<category>_article.md

<category>_evidence.json

category_summaries_index.json ← the Streamlit app reads this

Streamlit Dashboard

What it shows:

Overview: clusters & top terms (from Task 2)

Category Insights: sentiment distributions + generated summary (from Task 3)

Live Classifier: type/paste a review → model predicts sentiment (Task 1)

Run the app:

streamlit run app.py


In the sidebar, set:

Artifacts directory: your folder with Task 2 outputs

Summaries directory: .../task3_summaries

Sentiment model dir: either your LoRA adapter folder or merged_model/ (see next section)

On Windows, prefer raw strings or forward slashes for paths:

r"F:\MS\...\LLM_Customer_review_project"

"F:/MS/.../LLM_Customer_review_project"

Troubleshooting
1) PEFT “corda_config” or adapter load errors

PEFT versions can differ between Colab and local. Two fixes:

A. Sanitize adapter_config.json (quick)
Remove unknown keys like corda_config, eva_config, loftq_config, etc. Keep a minimal config:

{
  "base_model_name_or_path": "nlptown/bert-base-multilingual-uncased-sentiment",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "peft_type": "LORA",
  "r": 8,
  "target_modules": ["query", "value"],
  "task_type": "SEQ_CLS",
  "modules_to_save": ["classifier","score"]
}


B. Export a merged model (robust, PEFT-free)
In the environment where you trained Task 1:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

BASE_CKPT = "nlptown/bert-base-multilingual-uncased-sentiment"
SAVE_DIR  = r"/content/drive/MyDrive/Ironhack Assignments/LLM_Customer_review_project"

tok   = AutoTokenizer.from_pretrained(SAVE_DIR)
base  = AutoModelForSequenceClassification.from_pretrained(BASE_CKPT, num_labels=3, ignore_mismatched_sizes=True)
lora  = PeftModel.from_pretrained(base, SAVE_DIR)
merged = lora.merge_and_unload()

MERGED_DIR = os.path.join(SAVE_DIR, "merged_model")
os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR)
tok.save_pretrained(MERGED_DIR)
print("Saved merged model to:", MERGED_DIR)


Copy merged_model/ locally and point the Streamlit Sentiment model dir to it.

2) Windows path issues: \t becomes a tab

Use raw strings or forward slashes:

r"F:\path\with\backslashes"   # good
"F:/path/with/forward/slashes" # good
"F:\path\with\task3"           # BAD (\t is a tab)

3) “No generated article found for this category”

Ensure task3_summaries/category_summaries_index.json exists and contains entries for your category labels.

If category names changed (e.g., you reran Task 2), rerun Task 3 to regenerate summaries for the new labels.

Notes on Reproducibility

Random seeds: Set in Task 1 training; clustering uses deterministic seed for KMeans.

Versions pinned: See requirements.txt below.

Artifacts directory: Keep Task 2 and Task 3 outputs together. If you regenerate clusters, regenerate summaries.
