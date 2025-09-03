import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# load & clean column names
df = pd.read_excel("lab 3'/Lab Session Data.xlsx", sheet_name=2)
df.columns = df.columns.str.strip()

# pick the columns we need
cols = ['TSH', 'T3', 'TT4', 'Condition']
data = df[cols].copy()

# convert the three lab values to numeric, coerce errors → NaN
for c in ['TSH', 'T3', 'TT4']:
    data[c] = pd.to_numeric(data[c], errors='coerce')

# drop any row that still has NaN in those columns
data = data.dropna(subset=['TSH', 'T3', 'TT4'])

# create the binary label
data['label'] = np.where(data['Condition'] == 'NO CONDITION', 'NO', 'F')
#in this NO meaning no thyroid condition
#and F meaning there is hypothyroid condition


# question a1
# inter-distance between two class (in this code i have taken only NO and F as the two unique binary class)
# ---- centroid analysis ----
NO = data[data.label=='NO'][['TSH','T3','TT4']].values.astype(float)
F  = data[data.label=='F' ][['TSH','T3','TT4']].values.astype(float)

cent_NO = NO.mean(axis=0)
cent_F  = F.mean(axis=0)

distance = np.linalg.norm(cent_NO - cent_F)
std_NO  = NO.std(axis=0)
std_F   = F.std(axis=0)

print('centroid distance:', distance)
print('std NO:', std_NO)
print('std F :', std_F)


# question a2
# printting an histogram for patient having thyroid and using TSH as the main column for catograsing the dataset
counts, bins = np.histogram(data['TSH'], bins=20)
plt.hist(data['TSH'], bins=20)
mu, var = data['TSH'].mean(), data['TSH'].var()
plt.show()   # added to display this figure

# question a3
# printing the distance between v1 and v2 where r is the order of distance using manhattan(ord = 1), euclidean(ord = 2) and minkowshi(ord = 3 to 10)
v1, v2 = data.iloc[0][['TSH','T3','TT4']], data.iloc[1][['TSH','T3','TT4']] # three-dimensional numeric vector
dist = [np.linalg.norm(v1-v2, ord=r) for r in range(1,11)]
plt.plot(range(1,11), dist)
plt.show()   # added to display this figure

# question a4
# training and testing the data as train and test at the ratio of 7:3 for training and testing
X = data[['TSH','T3','TT4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# question a5 and a6
# training model to calculate the accuracy of three nearest neighburs
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)  #cals and gets only 3 neighrest neighbour
acc_train = clf.score(X_train, y_train)
acc_test  = clf.score(X_test, y_test)
# ---- results for A5/A6 ----
print('\n===== A5/A6  k-NN (k=3) =====')
print('Train accuracy :', acc_train)
print('Test  accuracy :', acc_test)


#question a7
pred = clf.predict(X_test) #returns a value (NO, F) which is of same length as X_train and y_train


#question a8
ks = range(1,12)
#returns a new knn classifier that trains and tests the accuracy of 11 k test and train cases
test_scores = [KNeighborsClassifier(k).fit(X_train,y_train).score(X_test,y_test) for k in ks]
plt.plot(ks, test_scores)
plt.show()   # added to display this figure

#question a9
cm = confusion_matrix(y_test, pred, labels=['NO','F'])
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average=None, labels=['NO','F'])
# ---- results for A9  ----
print('\n===== A9  Confusion Matrix & Metrics =====')
print('Confusion matrix:\n', cm)
print('Class order      : ["NO", "F"]')
print('Precision (NO, F):', prec)
print('Recall    (NO, F):', rec)
print('F1-score  (NO, F):', f1)
