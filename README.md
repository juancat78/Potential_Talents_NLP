# 🚀 Potential Talents - NLP Embedding Evaluation Project

This project explores how different **Natural Language Processing (NLP)** models perform in matching job candidates based on the semantic similarity between their job titles and a target job query.

## 🎯 Goal

To assess and compare the effectiveness of various embedding models (Word2Vec, GloVe, SBERT) in retrieving the most relevant candidates for a given job description, such as "human resources manager".

---

## 🛠️ Tools & Technologies

- Python
- Google Colab / Jupyter Notebooks
- Pandas, NumPy
- Scikit-learn
- Gensim (for Word2Vec and GloVe)
- Sentence-Transformers (SBERT)
- Evaluation Metrics: `Precision@K`, `nDCG@K`

---

## 🧪 Workflow

1. **Data Loading & Preprocessing:**  
   A dataset of candidate job titles is loaded and cleaned.

2. **Text Embedding Generation:**  
   - **Word2Vec:** trained using Gensim.
   - **GloVe:** pre-trained vectors.
   - **SBERT:** using `all-MiniLM-L6-v2`.

3. **Semantic Similarity Calculation:**  
   Cosine similarity is computed between the query and all job titles.

4. **Ranking & Manual Labeling:**  
   Top 10 candidates per model are manually labeled with `relevance = 1` or `0`.

5. **Evaluation Metrics:**  
   Models are compared using:
   - `Precision@10`
   - `nDCG@10` (Normalized Discounted Cumulative Gain)

---

## 📊 Sample Results

| Embedding Model | Precision@10 | nDCG@10 |
|------------------|--------------|---------|
| Word2Vec         | 0.50         | 0.79    |
| GloVe            | 0.60         | 0.75    |
| SBERT            | 0.20         | 1.00    |

> 🔍 **Note:** SBERT reached a perfect `nDCG@10` due to ideal ranking, but its `Precision@10` was lower, likely because it did not penalize terms like "Aspiring" as non-relevant.

---

## 🧠 Key Insights

- SBERT is effective for ranking due to its deep semantic understanding.
- Simpler models like Word2Vec and GloVe can outperform in precision under certain assumptions.
- Manual relevance labeling is crucial for realistic evaluation.

---

## 📁 Main Files

- `Potential_Talents_Embedding_Evaluation.ipynb`: Main notebook.
- `relevance_metrics.py`: Script with evaluation metric functions.
- `data/job_titles.csv`: Candidate job titles dataset.

---

## 👨‍💻 Author

Juan Pablo  
Project developed as part of the **AI Residency** at Apziva (2025).

---
