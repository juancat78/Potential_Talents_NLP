# NLP Talent Matching: Potential Talents

## Project Overview

This project focuses on applying Natural Language Processing (NLP) techniques to match job titles in the recruitment process. The main objective is to evaluate various text embedding models and their performance in ranking job titles based on their semantic similarity to a given query. The models tested include Word2Vec, GloVe, FastText, and SBERT, which are commonly used in NLP for tasks such as text similarity, recommendation, and search.

For each algorithm, a re-ranking strategy was implemented that leverages a set of “star candidates” — manually or semi-automatically selected job titles considered highly relevant. The final ranking score for each candidate combines two components: 1) Similarity to the original query (e.g., "human resources manager") and 2) Weighted similarity to the star candidates.

This approach refines the ranking by giving more weight to titles aligned with both the user query and the previously identified ideal matches. After re-ranking, a filtering step was applied based on a defined similarity threshold (cut-off point) to exclude titles below the relevance standard.

## Methodology

The job title matching task was approached through a multi-step process involving data preprocessing, embedding generation, similarity computation, re-ranking, and evaluation. The primary goal was to assess the performance of various embedding models in ranking job titles based on their semantic similarity to a given query, such as "human resources manager".

1. Data Preprocessing
To ensure consistency and improve the quality of the embeddings, the following preprocessing steps were applied to all job titles and queries:

Lowercasing: Converted all text to lowercase to maintain uniformity.

Punctuation Removal: Eliminated punctuation marks to focus on meaningful words.

Stopword Removal: Removed common stopwords that do not contribute to semantic meaning.

Lemmatization: Reduced words to their base or dictionary form to normalize variations.

These steps help in reducing noise and improving the semantic representation of the text.

2. Embedding Generation
Four embedding models were utilized to transform job titles and queries into vector representations:

Word2Vec: Utilized the Skip-gram model to generate word embeddings by predicting surrounding words in a context window. For multi-word job titles, embeddings were obtained by averaging the vectors of individual words.

GloVe: Generated word embeddings based on global word co-occurrence statistics from a large corpus. Similar to Word2Vec, multi-word titles were represented by averaging their constituent word vectors.

FastText: Extended Word2Vec by incorporating subword information, allowing it to handle rare and misspelled words effectively. Embeddings for job titles were computed by averaging the vectors of words and their subword components.

SBERT (Sentence-BERT): Produced sentence-level embeddings by fine-tuning BERT using a siamese network architecture. This approach captures the semantic meaning of entire sentences or phrases, making it suitable for job titles.

3. Similarity Computation
To measure the similarity between the query and each job title, cosine similarity was employed across all embedding models. This metric calculates the cosine of the angle between two vectors, providing a measure of their directional alignment and, consequently, their semantic similarity.

4. Re-ranking with Star Candidates
To enhance the relevance of the top-ranked job titles, a re-ranking strategy was implemented using a set of "star candidates"—job titles identified as highly relevant to the query. The re-ranking process involved:

Similarity Calculation: Computed the cosine similarity between each job title and the set of star candidates.

Weighted Scoring: Combined the similarity scores with the original query similarity, applying a weighting scheme to balance the influence of the query and the star candidates.

Final Ranking: Sorted the job titles based on the combined scores to prioritize those aligning closely with both the query and the star candidates.

5. Filtering with Cut-off Threshold
To ensure the quality of the recommended job titles, a filtering step was applied using a predefined similarity cut-off threshold. Job titles with combined similarity scores below this threshold were excluded from the final results, ensuring that only the most relevant titles were presented.

6. Evaluation Metrics
The performance of each embedding model was evaluated using the following metrics:

Precision@10: The proportion of relevant job titles among the top 10 results.

nDCG@10 (Normalized Discounted Cumulative Gain): A metric that considers the position of relevant results in the ranking, assigning higher importance to those appearing earlier.

### Embedding Models Evaluation

- **Word2Vec**:
  - **Precision@10**: 0.400. This indicates that 4 out of the top 10 results were relevant, which is acceptable but not ideal for high-precision recommendation tasks.
  - **nDCG@10**: 0.594. This suggests that the most relevant job titles are not always in the highest positions of the ranking, indicating room for improvement in the ranking order.

- **GloVe**:
  - **nDCG@10**: 0.753. GloVe outperformed **Word2Vec** in ranking relevant titles, demonstrating a better ability to order job titles according to their relevance. This makes GloVe a strong candidate for ranking tasks in the recruitment context.

- **FastText** (Correlation with Word2Vec):
  - **Correlation**: 0.884. The similarity between **Word2Vec** and **FastText** is very high, with both models providing similar results in terms of job title relevance. FastText’s advantage lies in its ability to model subword information, which makes it more resilient to spelling errors or rare job titles.
  
- **SBERT**:
  - **Precision@10**: 0.200. Only 2 out of the top 10 results were relevant. While **SBERT** correctly ranks highly relevant job titles at the top, the remaining results were not as useful. This suggests that **SBERT** is effective at recognizing the general meaning but struggles with specific relevance criteria (e.g., the term "aspiring").
  - **nDCG@10**: 1.000. The model perfectly ranked the most relevant job titles first, but this high nDCG score also reflects the model's inherent tendency to place even less relevant titles within the top ranks.

### Observations

- **Word2Vec** and **GloVe** performed better in terms of precision, with **GloVe** consistently outperforming **Word2Vec** in ranking tasks. **FastText** showed a high correlation with **Word2Vec**, suggesting similar ranking behavior, but with the added advantage of handling misspelled or rare job titles.
  
- **SBERT**, while achieving a perfect nDCG score, demonstrated a **low Precision@10**. This is primarily due to its tendency to rank less relevant job titles (e.g., “aspiring” job titles) highly, as it is more focused on understanding the general semantic meaning of the query rather than strict relevance for the recruitment process.

- **FastText**’s ability to capture subword information makes it highly suitable for datasets with misspelled or rare job titles, although it was not directly evaluated using the metrics.

## Results Summary

### Sample Results:
- **Word2Vec**: The model showed moderate performance in terms of ranking job titles, with an acceptable Precision@10 but a lower nDCG@10.
- **GloVe**: Demonstrated better performance with a higher nDCG@10 compared to Word2Vec.
- **FastText**: Although not evaluated with metrics, the correlation between FastText and Word2Vec was very high (0.884), indicating similar ranking behavior.
- **SBERT**: Achieved a perfect nDCG@10 but with low Precision@10, highlighting its strengths in semantic understanding but weaknesses in strict relevance alignment.

## Conclusions

- **GloVe** and **Word2Vec** are effective for ranking job titles in terms of both relevance and precision, with **GloVe** offering a slightly better ranking performance.
  
- **SBERT**, In this project, the pre-trained 'all-MiniLM-L6-v2' model from SBERT was used, which is optimized for general-purpose semantic similarity tasks in English. However, results suggest that this model may not have been trained on sufficiently representative examples of certain domain-specific or uncommon terms found in the dataset, such as non-standardized or contextually ambiguous job titles (e.g., titles including qualifiers like "aspiring"). This may explain its lower performance in terms of precision, despite achieving a high nDCG@10 score. Therefore, it would be advisable to explore alternative checkpoints that are better suited to the recruitment domain or to fine-tune the model on a domain-specific corpus to improve its alignment with stricter relevance criteria.
  
- **FastText** is particularly useful for datasets with rare or misspelled job titles due to its subword-level modeling, though direct evaluation metrics were not available.

## Main Files

#Potential_Talents.ipynb: Jupyter Notebook containing the full implementation of the project, including data preprocessing, embedding model comparisons, similarity computations, re-ranking logic with "star candidates", and evaluation using metrics such as Precision@10 and nDCG@10.

README.txt: This document provides a comprehensive overview of the project, including its objectives, methodologies (embedding models used, similarity comparison via cosine distance, re-ranking approach, and filtering), key results, and instructions for setup and execution.

Note: The original dataset used in this project is not included in the repository due to confidentiality agreements.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/juancat78/Potential_Talents_NLP.git
    cd Potential_Talents_NLP
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook:
    - Open `notebook.ipynb` in Jupyter or Google Colab to view and run the code.

---
