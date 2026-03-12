## Explainable Travel Buddy Recommender System

An experimental recommender system that suggests compatible travel buddies based on demographics, interests, and lifestyle preferences. The system is implemented in a single Jupyter notebook (`Recommender_System_Models.ipynb`) and demonstrates how to compute similarity between travelers and surface the most compatible matches.

### 1. Project Overview

- **Goal**: Recommend similar travelers (potential travel buddies) using structured profile data such as age, profession, hobbies, spoken languages, and lifestyle attributes (e.g. smoking, alcohol use, favorite colors, zodiac sign).
- **Core idea**: Turn each user profile into a feature representation and compute pairwise similarity scores; the top-scoring users are recommended as potential buddies.
- **Explainability**: Since the model is based on transparent text/vector features (e.g. `I_am_working_in_field`, `spoken_languages`, `hobbies`, etc.), you can inspect which attributes contribute to similarity and view similarity scores for each recommended traveler.

### 2. Main Components

- **Notebook**: `Recommender_System_Models.ipynb`  
  Contains all code for:
  - Loading and cleaning the dataset (`dataset.csv`)
  - Feature extraction with `CountVectorizer` and basic preprocessing
  - Computing cosine similarity for multiple profile fields
  - Combining similarity scores (including an age-based Euclidean distance component)
  - A `recommend_count(id, cosine_sim)` function that prints the selected user, their similarity scores, and the top similar travelers.

- **Dataset**: `dataset.csv` (expected)  
  A CSV file where each row represents a user, with (at least) the following columns:
  - `user_id`
  - `age`
  - `I_am_working_in_field`
  - `spoken_languages`
  - `hobbies`
  - `completed_level_of_education`
  - `favorite_color`
  - `relation_to_smoking`
  - `relation_to_alcohol`
  - `sign_in_zodiac`
  - `marital_status`
  - `I_like_movies`
  - `I_like_music`
  - `my_active_sports`
  - `profession`

> **Note**: Make sure `dataset.csv` is in the same directory as the notebook, or update the path in the notebook accordingly.

### 3. Environment & Dependencies

This project uses standard scientific Python libraries. A typical environment includes:

- Python 3.8+  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  
- `nltk`  
- `gensim`  
- `Pillow` (`PIL`)  
- `requests`

You can install them with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk gensim pillow requests
```

The notebook also downloads NLTK stopwords at runtime:

```python
import nltk
nltk.download("stopwords")
```

### 4. How to Run the Notebook

1. **Clone or copy** this project folder to your machine.
2. Ensure `Recommender_System_Models.ipynb` and `dataset.csv` are in the same directory.
3. (Optional but recommended) **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -U pip
   pip install numpy pandas scikit-learn matplotlib seaborn nltk gensim pillow requests
   ```

4. **Start Jupyter** (or open in another notebook environment):

   ```bash
   pip install notebook
   jupyter notebook
   ```

5. Open `Recommender_System_Models.ipynb` and run the cells in order (Kernel → Restart & Run All).

### 5. Using the Recommender

Inside the notebook, after running all cells, you can call:

```python
recommend_count(144)
```

Where `144` is an example `user_id`. Replace it with any valid `user_id` from `dataset.csv`. The function will:

- Display the selected user’s profile
- Print similarity scores to other travelers
- Display the top recommended travel buddies (by similarity)

### 6. Extending the Project

Ideas for further work:

- **Better explainability**:  
  - Break down similarity by feature group (e.g. hobbies vs. languages vs. age).  
  - Visualize contributions of each attribute to the final similarity score.
- **Model improvements**:  
  - Use TF–IDF or word embeddings instead of simple counts.  
  - Tune weights for different profile fields (e.g. give more weight to hobbies or age).
- **Interface**:  
  - Wrap the notebook logic in a Python module or API.  
  - Build a simple web UI where a user can submit or select a profile and see recommended buddies plus explanations.

### 7. License & Citation

If you use this project or build on it in academic or industrial work, please cite it as:

> Explainable Travel Buddy Recommender System (Jupyter notebook implementation). Unpublished project.

