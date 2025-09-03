# ------------------------------------------------------------
# Lab-06 | Decision-Tree utilities on Final_ML_Dataset.xlsx
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 0. File & basic setup
# --------------------------------------------------

FILE = r"C:\Users\aspav\Downloads\Final_ML_Dataset.xlsx"  # Path to dataset (adjust as needed)
TARGET_COL = "Label"  # Target column for classification

# Read the Excel file
df = pd.read_excel(FILE)

# Drop rows where target is NaN to ensure clean data
df = pd.read_excel(FILE).dropna(subset=[TARGET_COL])

# Split into features (X) and target (y)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -------------------------------------------------
# A1 Entropy (with optional equal-width binning)
# -------------------------------------------------

def entropy(series, bins=None):
    # Calculate the entropy of a pandas Series.
    # If bins are provided, first bin the data into categorical bins.
    if bins:
        # Cut the continuous numeric data into equal width bins, label them 0,1,2,...
        series = pd.cut(series, bins=bins, labels=False)
    # Get the probability distribution of different categories/values
    probs = series.value_counts(normalize=True)
    # Apply the entropy formula ignoring zero probabilities
    return -sum(p * math.log2(p) for p in probs if p > 0)

# If target y is numeric, bin into 4 equal-width bins for entropy calculation
if y.dtype in ["float64", "int64"]:
    y_binned = pd.cut(y, bins=4, labels=False)
else:
    y_binned = y

print("Entropy of target:", entropy(y_binned))
# -------------------------------------------------
# A2 Gini index
# -------------------------------------------------

def gini(series, bins=None):
    # Calculate the Gini index of a pandas Series.
    if bins:
        series = pd.cut(series, bins=bins, labels=False)
    probs = series.value_counts(normalize=True)
    # Apply Gini index formula: 1 - sum of squared probabilities
    return 1 - sum(p**2 for p in probs)

print("Gini of target:", gini(y_binned))

# -------------------------------------------------
# A3/A4 Information-gain splitter for root node
# -------------------------------------------------

def information_gain(parent, left, right):
    # Calculate information gain from splitting `parent` into `left` and `right`
    def ent(s):
        return entropy(s)
    N = len(parent)
    # IG = entropy(parent) - weighted_average_entropy(children)
    return ent(parent) - (len(left) / N * ent(left) + len(right) / N * ent(right))

def find_root_feature(X, y, max_bins=4):
    # Find the best feature and threshold (value) to split the root node based on max info gain
    best_gain, best_feat = -1, None

    for col in X.columns:
        col_vals = X[col]

        # If column is numeric, bin it to convert into categorical
        if col_vals.dtype in ["float64", "int64"]:
            col_vals = pd.cut(col_vals, bins=max_bins, labels=False)
        else:
            # For categorical data, convert labels to numeric codes
            col_vals = col_vals.astype('category').cat.codes

        unique = col_vals.unique()

        # Try splitting on each unique binned/category value
        for val in unique:
            mask = col_vals <= val  # Left split: data where feature <= val
            gain = information_gain(y, y[mask], y[~mask])

            if gain > best_gain:
                best_gain, best_feat = gain, (col, val)

    return best_feat, best_gain

root_feat, root_gain = find_root_feature(X, y)
print("Root feature & threshold chosen:", root_feat, "Gain:", root_gain)

# -------------------------------------------------
# A5 Build a full Decision-Tree with sklearn
# -------------------------------------------------

# Before feeding into sklearn, encode all feature columns as numeric
X_enc = X.copy()
for col in X_enc.columns:
    X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

# Initialize and train Decision Tree classifier (no max depth limit)
clf = DecisionTreeClassifier(max_depth=None, random_state=42)
clf.fit(X_enc, y)

# ===== A6 Decision-tree plot =====

plt.figure(figsize=(20, 10))

# Plot the tree with feature names and class names
plot_tree(
    clf,
    feature_names=list(X_enc.columns),
    class_names=[str(c) for c in clf.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree – Final_ML_Dataset")
plt.show()

# ===== A7 2-D decision boundary (first two numeric columns) =====

# Select first two numeric columns for 2D plotting
num_cols = X_enc.select_dtypes(include=np.number).columns[:2].tolist()

if len(num_cols) == 2:
    X2 = X_enc[num_cols]

    # Train a smaller decision tree limited to depth 4 for clear boundary visualization
    clf2 = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X2, y)

    # Create mesh grid covering feature space
    x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
    y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict classes over grid points
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot decision boundary contour with scatter points of original data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X2.iloc[:, 0], y=X2.iloc[:, 1], hue=y, palette='Set2')
    plt.contourf(xx, yy, Z, alpha=0.2, levels=np.unique(Z), cmap=plt.cm.Paired)

    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.title("2-D Decision Boundary – Final_ML_Dataset")
    plt.show()
else:
    print("Need at least two numeric columns for 2-D boundary plot.")

