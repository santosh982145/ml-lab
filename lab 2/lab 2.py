import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


########################################################################
# GLOBAL CONFIGURATION
########################################################################
EXCEL_FILE = r"C:\Users\aspav\OneDrive\Documents\final-ml\ml-lab\lab 2\Lab Session Data.xlsx"
# Path to the Excel file containing all data sheets


########################################################################
# Q1 – Linear regression on Purchase data
########################################################################
print("=== Q1 ===")
data = pd.read_excel(EXCEL_FILE, sheet_name="Purchase data")

# Extract features matrix G (candies, mangoes, milk packets)
G = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()

# Extract target vector b (payment in Rs)
b = data["Payment (Rs)"].values.reshape(-1, 1)

# Get dimensions and rank of feature matrix
rows, cols = G.shape
rank_G = np.linalg.matrix_rank(G)

# Calculate regression weights w using pseudo-inverse (least squares)
w = np.linalg.pinv(G) @ b

print(f"Rows: {rows}")  # Number of data samples
print(f"Cols: {cols}")  # Number of features
print(f"Rank: {rank_G}")  # Rank of feature matrix (dimensionality)
print(f"Vector-space dim: {rank_G}")
print(f"Binary combos (2^n): {2 ** cols}")  # Possible binary combinations if interpreted categorically
print(f"Candy Rs {w[0,0]:.2f}")  # Coefficient price per candy
print(f"Mango Rs {w[1,0]:.2f}")  # Coefficient price per kg mango
print(f"Milk Rs {w[2,0]:.2f}\n")  # Coefficient price per milk packet


########################################################################
# Q2 – Logistic classifier (rich vs poor) on Purchase data
########################################################################
print("=== Q2 ===")
sheet = pd.read_excel(EXCEL_FILE, sheet_name="Purchase data")

# Features matrix same as before
G = sheet[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values

# Binary labels: 1 if Payment > 200 (rich), else 0 (poor)
z = (sheet["Payment (Rs)"] > 200).astype(int).values.reshape(-1, 1)

# Add intercept term (column of ones) to features for bias
G_aug = np.hstack([np.ones((G.shape[0], 1)), G])

# Logistic regression weights using Least Squares approximation (not typical logistic optimization)
theta = np.linalg.pinv(G_aug.T @ G_aug) @ G_aug.T @ z

# Predict class labels as 1 if predicted value >= 0.5, else 0
pred = (G_aug @ theta >= 0.5).astype(int)

# Calculate accuracy score
score = (pred == z).mean()

# Add columns to show actual vs guessed label as string
sheet["Actual"] = np.where(z, "RICH", "POOR")
sheet["Guess"] = np.where(pred, "RICH", "POOR")

print(sheet[["Payment (Rs)", "Actual", "Guess"]])
print("Accuracy:", round(score, 2))
print("Params:", theta.flatten(), "\n")


########################################################################
# Q3 – IRCTC stock summary statistics
########################################################################
print("=== Q3 ===")
tbl = pd.read_excel(EXCEL_FILE, sheet_name="IRCTC Stock Price")
price = tbl["Price"]

# Compute mean and variance of price
mu = price.mean()
var = price.var(ddof=0)  # Population variance (ddof=0)

# Mean price on Wednesdays
wed = tbl.loc[tbl["Day"] == "Wed", "Price"].mean()

# Mean price in April
apr = tbl.loc[tbl["Month"] == "Apr", "Price"].mean()

# Proportion of days with a price loss (negative change %)
loss = (tbl["Chg%"] < 0).mean()

# Proportion of Wednesdays with a price gain
gain = ((tbl["Chg%"] > 0) & (tbl["Day"] == "Wed")).sum() / (tbl["Day"] == "Wed").sum()

# Print rounded statistics
print(mu.round(2))
print(var.round(2))
print(wed.round(2))
print(apr.round(2))
print(loss.round(4))
print(gain.round(4), "\n")


########################################################################
# Q4 – Binary similarity (first two rows) thyroid data
########################################################################
print("=== Q4 ===")
tbl = pd.read_excel(EXCEL_FILE, sheet_name="thyroid0387_UCI")

# Identify columns that have only 'f' or 't' (binary flags)
b = [c for c in tbl.columns if set(tbl[c].dropna()) == {'f', 't'}]

# Convert first and second row binary columns to 1/0 arrays
x = (tbl.iloc[0][b] == 't').astype(int).values
y = (tbl.iloc[1][b] == 't').astype(int).values

# Compute contingency counts for Jaccard and SMC
a = (x & y).sum()     # both true
b_ = (x & ~y).sum()  # x true, y false
c = (~x & y).sum()   # x false, y true
d = (~x & ~y).sum()  # both false

# Jaccard Coefficient = |X ∩ Y| / |X ∪ Y|
jc = a / (a + b_ + c) if a + b_ + c else None

# Simple Matching Coefficient (SMC) = (matches) / (total)
smc = (a + d) / (a + b_ + c + d) if a + b_ + c + d else None

print("Binary:", b)
print("JC:", round(jc, 4))
print("SMC:", round(smc, 4), "\n")


########################################################################
# Q5 – Same as Q4 (already computed)
########################################################################
print("=== Q5 ===")
print("Same as Q4 output, skipping duplicate print.\n")


########################################################################
# Q6 – Cosine similarity between first two numeric rows
########################################################################
print("=== Q6 ===")
sheet = pd.read_excel(EXCEL_FILE, sheet_name="thyroid0387_UCI")

# Convert values to floats where possible, else 0.0
def floatize(vals):
    return np.array([float(v) if str(v).lstrip('+-').replace('.', '', 1).isdigit() else 0.0 for v in vals])

u = floatize(sheet.iloc[0])
v = floatize(sheet.iloc[1])

# Compute cosine similarity = dot(u,v) / (||u|| * ||v||)
cos_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print(round(cos_sim, 4), "\n")


########################################################################
# Q7 – Heat-maps for Jaccard / SMC / Cosine on first 20 rows
########################################################################
print("=== Q7 ===")
frame = pd.read_excel(EXCEL_FILE, sheet_name="thyroid0387_UCI").iloc[:20]

# Columns with binary flags 'f'/'t'
flag = [c for c in frame.columns if set(frame[c].dropna()) <= {'f', 't'}]

# Convert binary to integer arrays
A = (frame[flag] == 't').astype(int).values

# Extract numeric columns and fill NaNs with 0
B = frame.select_dtypes(include='number').fillna(0).values

for tag, M in [('J', A), ('S', A), ('C', B)]:
    m = len(M)
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            p, q = M[i], M[j]
            if tag == 'C':
                # Cosine similarity
                H[i, j] = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12)
            else:
                # a,b,c,d counts for binary similarity
                a = (p & q).sum()
                b = (p & ~q).sum()
                c = (~p & q).sum()
                d = (~p & ~q).sum()
                if tag == 'J':
                    # Jaccard coefficient
                    H[i, j] = a / (a + b + c) if a + b + c else 0
                else:  # SMC
                    H[i, j] = (a + d) / (a + b + c + d) if a + b + c + d else 0

    sns.heatmap(H, cmap='viridis', cbar=False)
    plt.title(tag)
    plt.show()


########################################################################
# Q8 – Missing-value imputation (median / mode)
########################################################################
print("=== Q8 ===")
frame = pd.read_excel(EXCEL_FILE, sheet_name="thyroid0387_UCI")

for c in frame.columns:
    # Convert column values to numeric where possible, '?' converted to NaN
    temp = pd.to_numeric(frame[c].astype(str).replace('?', np.nan), errors='coerce')
    # Use median for numeric columns, mode for categorical
    filler = temp.median() if pd.api.types.is_numeric_dtype(temp) else temp.mode()[0]
    # Fill missing values with filler
    frame[c] = temp.fillna(filler)

print(frame.isnull().sum(), "\n")  # Confirm no missing values remain


########################################################################
# Q9 – Min-max normalisation (0-1 scaling)
########################################################################
print("=== Q9 ===")
frame = pd.read_excel(EXCEL_FILE, sheet_name="thyroid0387_UCI")

for col in frame.select_dtypes(include=np.number):
    low, high = frame[col].min(), frame[col].max()
    # Scale to [0,1] range; if all values identical (high==low) set column to zero
    frame[col] = (frame[col] - low) / (high - low) if high > low else 0

print(frame.head())  # Show normalized data head

