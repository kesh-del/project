Fake News Detection Project

This project implements a machine learning pipeline to detect fake news using text classification. It compares the performance of three models: Multinomial Naive Bayes, Logistic Regression, and Passive Aggressive Classifier.

Prerequisites:
- Python 3.12.4
- Required libraries: numpy, pandas, scikit-learn, matplotlib

Dataset:
- The dataset (fake_news_dataset.csv) should be placed in one of the following paths:
  - C:\Users\DELL\Downloads\archive (1)\fake_news_dataset.csv
  - C:\Users\DELL\Downloads\fake_news_dataset.csv
  - C:\Users\DELL\Downloads\archive\fake_news_dataset.csv
- Expected columns: 'title', 'text', 'date', 'source', 'author', 'category', 'label' (where 'label' is 'real' or 'fake')

File:
- fakenew.ipynb: Jupyter notebook containing the complete code for loading the dataset, preprocessing, model training, evaluation, and visualization.

Usage:
1. Ensure the dataset is in one of the specified paths.
2. Install dependencies: pip install numpy pandas scikit-learn matplotlib
3. Run the Jupyter notebook (fakenew.ipynb) to execute the pipeline.
4. The script will:
   - Load and preprocess the dataset
   - Train three models
   - Output accuracy, precision, recall, F1 score, and confusion matrices
   - Display a bar chart comparing model performance

Output:
- Console output includes dataset details, missing values, and model performance metrics.
- A bar chart visualizes the comparison of Accuracy, Precision, Recall, and F1 Score across models.

Notes:
- The dataset is preprocessed by filling missing values with empty strings and combining 'title', 'author', and 'text' into a single feature.
- TF-IDF vectorization is used for feature extraction with English stop words removed and max_df=0.7.
- Models are evaluated on a 20% validation set (random_state=42).

For issues or contributions, please open a pull request or issue on the GitHub repository.
