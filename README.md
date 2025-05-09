# NLP Talent Matching: Potential Talents

## Project Overview

This project focuses on applying Natural Language Processing (NLP) techniques to match job titles in the recruitment process. The main objective is to evaluate various text embedding models and their performance in ranking job titles based on their semantic similarity to a given query. The models tested include **Word2Vec**, **GloVe**, **FastText**, and **SBERT**, which are commonly used in NLP for tasks such as text similarity, recommendation, and search.

## Key Techniques

1. **Word2Vec**: A shallow neural network model used to learn word associations from a large corpus of text, capturing semantic meaning by considering the surrounding context of words.
2. **GloVe**: A model that generates word embeddings based on matrix factorization techniques, focusing on capturing global co-occurrence statistics from large corpora.
3. **FastText**: Similar to Word2Vec but with the added ability to handle subword information, which allows it to better capture rare and misspelled words.
4. **SBERT (Sentence-BERT)**: A modification of the BERT architecture, which produces sentence-level embeddings and is particularly effective for tasks that require understanding the meaning of entire sentences or phrases.

## Methodology

The job title matching task was approached by comparing the performance of the four embedding models in terms of their ability to rank job titles based on their relevance to a given query, **"human resources manager"**. The models were evaluated using two primary metrics:

- **Precision@10**: The fraction of relevant job titles in the top 10 results.
- **nDCG@10 (Normalized Discounted Cumulative Gain)**: A ranking metric that accounts for the position of relevant results, where higher values indicate better-ranked relevance.

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

## Conclusion

- **GloVe** and **Word2Vec** are effective for ranking job titles in terms of both relevance and precision, with **GloVe** offering a slightly better ranking performance.
- **SBERT**, while great for general semantic understanding, requires fine-tuning or domain-specific adjustments to improve precision for specific tasks like job title matching.
- **FastText** is particularly useful for datasets with rare or misspelled job titles due to its subword-level modeling, though direct evaluation metrics were not available.

## Main Files
- **`notebook.ipynb`**: Jupyter Notebook containing the code and methodology used for this project.
- **`requirements.txt`**: The dependencies required to run the notebook.
  
(Note: The original dataset in `.csv` format is not included due to confidentiality agreements.)

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

Este **README** está actualizado para reflejar los resultados y comparaciones correctas entre los diferentes métodos de **embedding**, así como las observaciones clave de los modelos evaluados. También he incluido la nota de confidencialidad sobre el archivo `.csv`.
