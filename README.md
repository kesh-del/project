Fake News Detection Project

This project implements a machine learning pipeline to detect fake news using text classification. It compares Multinomial Naive Bayes, Logistic Regression, and Passive Aggressive Classifier models, and includes a Streamlit web interface for real-time predictions.

Prerequisites:
- Python 3.12.4
- Required libraries: numpy, pandas, scikit-learn, matplotlib, nltk, streamlit, joblib
- Install dependencies: pip install numpy pandas scikit-learn matplotlib nltk streamlit joblib

Setup Instructions:
1. Create and activate a virtual environment (recommended):
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
2. Install dependencies: pip install numpy pandas scikit-learn matplotlib nltk streamlit joblib
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

Dataset:
- Place fake_news_dataset.csv in one of these paths:
  - C:\Users\DELL\Downloads\archive (1)\fake_news_dataset.csv
  - C:\Users\DELL\Downloads\fake_news_dataset.csv
  - C:\Users\DELL\Downloads\archive\fake_news_dataset.csv
- Expected columns: 'title', 'text', 'date', 'source', 'author', 'category', 'label' ('real' or 'fake')

Files:
- fakenew.ipynb: Jupyter notebook for data preprocessing, model training, evaluation, and visualization.
- app.py: Streamlit app for interactive fake news prediction.
- tfidf_vectorizer.pkl: Saved TF-IDF vectorizer.
- MultinomialNB.pkl, LogisticRegression.pkl, PassiveAggressiveClassifier.pkl: Saved trained models.

Usage:
1. Ensure the dataset is in a specified path.
2. Run the Jupyter notebook (fakenew.ipynb) to:
   - Preprocess data using NLTK (tokenization, lemmatization, stop word removal)
   - Train and evaluate models
   - Save models and vectorizer
   - Output metrics (accuracy, precision, recall, F1) and a comparison chart
3. Run the Streamlit app:
   - Command: streamlit run app.py
   - Enter article title, author, and/or text in the web interface
   - View prediction (real/fake), confidence scores, and explanation based on key words

Output:
- Notebook: Prints dataset details, missing values, model metrics, and confusion matrices; displays a bar chart comparing model performance.
- Streamlit App: Shows prediction, confidence probabilities, and an explanation of influential words.

Notes:
- Data preprocessing includes NLTK-based cleaning and TF-IDF vectorization (max_df=0.7, English stop words).
- Models are evaluated on a 20% validation set (random_state=42).
- The Streamlit app uses the Logistic Regression model for predictions.

For issues or contributions, please open a pull request or issue on the GitHub repository.
