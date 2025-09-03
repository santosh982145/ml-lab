import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from scipy.stats import uniform


# =============================
# Load dataset
# =============================
DATA_PATH = r"lab 7\Final_ML_Dataset.xlsx"  # Path to dataset Excel file
data = pd.read_excel(DATA_PATH)  # Read dataset into pandas DataFrame


# =============================
# Safe preprocessing for text columns
# =============================
# Detect if 'Teacher' and/or 'Student' columns are present for text features
text_cols = [c for c in ['Teacher', 'Student'] if c in data.columns]

# Convert selected text columns to string type and fill any NaNs with empty string
if text_cols:
    data.loc[:, text_cols] = data[text_cols].astype(str).fillna('')

# Combine text columns into single 'text' feature for modeling
if set(text_cols) == {'Teacher', 'Student'}:
    data['text'] = data['Teacher'] + ' ' + data['Student']
elif 'Teacher' in text_cols:
    data['text'] = data['Teacher']
elif 'Student' in text_cols:
    data['text'] = data['Student']
else:
    raise ValueError("Expected at least one of ['Teacher','Student'] columns for text features.")


# =============================
# Label handling (coerce, drop NaNs, remap to 0..K-1)
# =============================
if 'Label' not in data.columns:
    raise ValueError("Missing 'Label' column in the dataset.")

# Convert label column to numeric; invalid values become NaN
data['Label'] = pd.to_numeric(data['Label'], errors='coerce')

# Count and drop rows with invalid/missing labels
missing_cnt = int(data['Label'].isna().sum())
if missing_cnt:
    print(f"Dropping {missing_cnt} rows with missing/invalid Label")
    data = data.dropna(subset=['Label']).copy()

# Map original labels to contiguous integers (0 ... K-1), needed for many classifiers
unique_classes = np.sort(data['Label'].unique())
class_map = {old: i for i, old in enumerate(unique_classes)}  # Map old labels to new indices
data['Label'] = data['Label'].map(class_map).astype(int)
print("Class mapping used:", class_map)


# =============================
# Train-test split
# =============================
X = data['text']  # Feature: combined text column
y = data['Label']  # Target labels
# Split dataset into training and testing sets with stratification on target label
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)


# =============================
# Define pipelines for classifiers
# =============================
# Common TF-IDF vectorizer setup for textual features
common_tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Dictionary of different classification pipelines
# Each pipeline chains TF-IDF vectorizer and a classifier
classifiers = {
    'SVM': Pipeline([
        ('tfidf', common_tfidf),
        ('clf', SVC(probability=True, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('tfidf', common_tfidf),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('tfidf', common_tfidf),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ]),
    'NaiveBayes': Pipeline([
        ('tfidf', common_tfidf),
        ('clf', MultinomialNB())
    ]),
    'MLP': Pipeline([
        ('tfidf', common_tfidf),
        ('clf', MLPClassifier(random_state=42, max_iter=300))
    ]),
}


# =============================
# Training and evaluating classifiers helper function
# =============================
def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    rows = []
    for name, pipeline in classifiers.items():
        pipeline.fit(X_train, y_train)  # Train classifier pipeline
        y_pred = pipeline.predict(X_test)  # Predict on test set

        # Calculate multiple metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # Collect metrics results
        rows.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

        # Display classification report per model
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
   
    # Return results as a DataFrame for comparison
    return pd.DataFrame(rows)


# =============================
# Evaluate classifiers without hyperparameter tuning
# =============================
results_df = evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test)
print("Summary of classification results:")
print(results_df)


# =============================
# Hyperparameter tuning: SVM pipeline
# =============================
# Define SVM pipeline again (separately) for RandomizedSearchCV
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', SVC(probability=True, random_state=42))
])

# Parameter distribution for searching:
# - C: regularization strength (continuous uniform distribution from 0.1 to 10.1)
# - gamma: kernel coefficient, discrete options
# - kernel: choice among linear, rbf, or polynomial kernels
param_distributions = {
    'clf__C': uniform(0.1, 10),
    'clf__gamma': ['scale', 'auto'],
    'clf__kernel': ['linear', 'rbf', 'poly']
}

# Setup RandomizedSearchCV for hyperparameter tuning with 5-fold cross-validation
random_search = RandomizedSearchCV(
    svm_pipeline,
    param_distributions,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

random_search.fit(X_train, y_train)  # Run search


print("Best hyperparameters for SVM:")
print(random_search.best_params_)


# Evaluate tuned SVM on test data
y_pred_svm = random_search.predict(X_test)

print("Classification report for tuned SVM:")
print(classification_report(y_test, y_pred_svm, zero_division=0))


# Append tuned SVM results to previous results DataFrame
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_svm, average='weighted', zero_division=0
)

tuned_row = pd.DataFrame([{
    'Model': 'SVM (Tuned)',
    'Accuracy': svm_acc,
    'Precision': svm_precision,
    'Recall': svm_recall,
    'F1 Score': svm_f1
}])

results_df = pd.concat([results_df, tuned_row], ignore_index=True)

print("Updated summary of classification results with tuned SVM:")
print(results_df)

