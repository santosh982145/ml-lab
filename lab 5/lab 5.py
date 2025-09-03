import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load the actual dataset
file_path = r"lab 5\Final_ML_Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Process actual transcript data
#used for extracting only one attribute
def extract_features(text):
    """Extract numerical features from viva responses"""
    if pd.isna(text):
        return 0, 0, 0
    text = str(text).lower()
    length = len(text.split())
    tech_terms = sum(1 for word in ['algorithm', 'model', 'regression', 'classification', 'accuracy'] if word in text) #attribute extarction
    confidence = 1 if any(word in text for word in ['sure', 'correct', 'definitely', 'yes']) else 0 #target
    return length, tech_terms, confidence

# Process data properly
features = []
labels = []

# Skip header row and process actual data
for idx, row in df.iterrows():
    if idx > 0:  # Skip header
        teacher_text = str(row['Teacher']) if pd.notna(row['Teacher']) else ""
        student_text = str(row['Student']) if pd.notna(row['Student']) else ""
       
        # Extract features from student responses (more relevant)
        student_len, student_tech, student_conf = extract_features(student_text)
       
        # Create features
        combined_features = [
            len(teacher_text.split()),  # Teacher response length
            student_len,                # Student response length
            student_tech,               # Technical terms in student response
            student_conf                # Confidence indicator
        ]
        features.append(combined_features)
       
        # Handle labels - use numeric values from Label column
        label = row['Label'] if pd.notna(row['Label']) and str(row['Label']).isnumeric() else 3
        labels.append(float(label))

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Ensure we have consistent data
min_samples = min(len(X), len(y))
X = X[:min_samples]
y = y[:min_samples]

print(f"Loaded {len(X)} samples with {X.shape[1]} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A1-A2: Linear Regression with one feature (student response length)
print("\n=== A1-A2: Linear Regression with one feature ===")
reg_one = LinearRegression()
reg_one.fit(X_train[:, [1]], y_train)  # Using student response length

y_train_pred_one = reg_one.predict(X_train[:, [1]])
y_test_pred_one = reg_one.predict(X_test[:, [1]])

mse_train = mean_squared_error(y_train, y_train_pred_one)
mse_test = mean_squared_error(y_test, y_test_pred_one)

r2_train = r2_score(y_train, y_train_pred_one)
r2_test = r2_score(y_test, y_test_pred_one)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

mape_train = mean_absolute_percentage_error(y_train, y_train_pred_one)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred_one)

print(f"Train - MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAPE: {mape_train:.2f}, R2: {r2_train:.2f}")
print(f"Test  - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}, R2: {r2_test:.2f}")

# A3: Linear Regression with all features
print("\n=== A3: Linear Regression with all features ===")
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)

y_train_pred_all = reg_all.predict(X_train)
y_test_pred_all = reg_all.predict(X_test)

mse_train_all = mean_squared_error(y_train, y_train_pred_all)
mse_test_all = mean_squared_error(y_test, y_test_pred_all)

r2_train_all = r2_score(y_train, y_train_pred_all)
r2_test_all = r2_score(y_test, y_test_pred_all)

rmse_train_all = np.sqrt(mse_train_all)
rmse_test_all = np.sqrt(mse_test_all)

mape_train_all = mean_absolute_percentage_error(y_train, y_train_pred_all)
mape_test_all = mean_absolute_percentage_error(y_test, y_test_pred_all)

print(f"Train - MSE: {mse_train_all:.2f}, RMSE: {rmse_train_all:.2f}, MAPE: {mape_train_all:.2f}, R2: {r2_train_all:.2f}")
print(f"Test  - MSE: {mse_test_all:.2f}, RMSE: {rmse_test_all:.2f}, MAPE: {mape_test_all:.2f}, R2: {r2_test_all:.2f}")


# A4-A5: K-Means Clustering
print("\n=== A4-A5: K-Means Clustering ===")
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_train)
labels = kmeans.labels_

silhouette = silhouette_score(X_train, labels)
ch_score = calinski_harabasz_score(X_train, labels)
db_score = davies_bouldin_score(X_train, labels)

print(f"Silhouette Score: {silhouette:.2f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")
print(f"Davies-Bouldin Score: {db_score:.2f}")


# A6
print("\n=== A6: Clustering Evaluation for Different k Values ===")
max_k = min(10, len(X_train) // 2)
k_values = range(2, max_k + 1)
silhouette_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)
    labels = kmeans.labels_
   
    silhouette_scores.append(silhouette_score(X_train, labels))
    ch_scores.append(calinski_harabasz_score(X_train, labels))
    db_scores.append(davies_bouldin_score(X_train, labels))

# Print the scores for each k value
print("k\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin")
for i, k in enumerate(k_values):
    print(f"{k}\t{silhouette_scores[i]:.3f}\t\t{ch_scores[i]:.3f}\t\t\t{db_scores[i]:.3f}")


# A7: Elbow Plot
print("\n=== A7: Elbow Plot ===")
distortions = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(distortions)+1), distortions, 'bo-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Plot for Optimal k')
plt.grid()
if len(distortions) >= 3:
    plt.axvline(x=3, color='r', linestyle='--', label='Optimal k=3')
    plt.legend()
plt.show()

print("\n=== LAB 5 COMPLETED ===")
print(f"Successfully processed {len(X)} transcript samples")
print(f"Features extracted: Teacher length, Student length, Technical terms, Confidence")
print("All tasks A1-A7 completed using actual Final_ML_Dataset.xlsx")
