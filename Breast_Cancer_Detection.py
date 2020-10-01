"""

* AUTOR: JOÃO VITOR MARTINS FERNANDEZ

* DATA: 10/08/2020 

* DESCRIPTION: This project were writed in Spyder IDE and use the breast 
  cancer dataset from Winsconsin University Repository. It models AIs
  algorithms to detect breast cancer. The models user were 'Naive-Bayes',
  'Decision Tree', 'K-Nearst Neigbors' (KNN), 'Logistic Regression',
  'Suport Vector Machine' (SVM) and 'Artificial Neural Networks' (ANN).

* METHODS: The results of confusion matrix, accuracy, precision, recall
  and f1-score are presented. The models are compared. 

  - Confusion matrix = returns the number of positive positives (PP),
    positive negatives (PN), false positives (FP) and false negatives (FN).
  - Accuracy = ratio of correctly predicted observation to the total observations. 
  - Precision = (FP)/(PP + PN)
  - Recall (Sensitivity) = (FN)/(PP + PN)
  - F1 score = Weighted average of Precision and Recall. 
    This score takes both false positives and false negatives

* RESULTS:The best values for accuracy are obtained for KNN and ANN,
  both with 0.9657142857142857.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Libraries for pre-processing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from pandas.plotting import scatter_matrix

# =============================================================================
# (I) - IMPORTING THE DATASET
# =============================================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['ID', 'Clump_thickness', 'Uniform_Cell_Size', 'Uniform_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv(url, names = names)

# =============================================================================
# (II) - PREPROCESSING DATASET WITH PANDAS DATAFRAME
# =============================================================================

# Correct missing '?' and inconsistent '-99999' values
df.replace('?', -99999, inplace = True)     
print(df.axes, "\n")

# Remove 'ID' Column 
df.drop(['ID'], 1, inplace = True)

# Print the shape and data type of the Dataset
print("\nDataset shape:", df.shape) 
print("\nDataset types:")
print(df.dtypes)  # Type - must be 'numeric' to further analysis

# Transform data to 'numeric'
# df = df.apply(pd.to_numeric)

# Describe the dataset
print("\nDataset parameters:")
print(df.describe())   # Global Statistic Vision

# To plot histograms for each variable
df.hist(figsize = (10, 10))
plt.show()

# To create scatter plot matrix
scatter_matrix(df, figsize = (18, 18))
plt.show()

# =============================================================================
# (III) SPLITTING DATASET INTO TRAINING AND TESTING SUBSETS
# =============================================================================

# To create datasets for training
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# For reproducibility
seed = 8

# P for predictor C for classifier datasets
P = np.array(df.drop(['Class'], 1))
C = np.array(df['Class'])

# Splitting: 0.75 for training, 0.25 for test
P_train, P_test, C_train, C_test = train_test_split(P, C, 
                                         test_size = 0.25, random_state = seed)

# =============================================================================
# (IV) BUILD THE MACHINE LEARNING MODEL
# =============================================================================

# Libraries for AI algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

scoring = 'accuracy'

# Defining AI models to train
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree', DecisionTreeClassifier(criterion='entropy', random_state = 0)))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 7)))
models.append(('Logistic Regression', LogisticRegression(random_state = 1, solver='lbfgs')))
models.append(('SVM', SVC()))
models.append(('ANN', MLPClassifier(verbose = True,
                                    max_iter=1000,
                                    tol = 0.0000010,
                                    solver = 'adam',
                                    hidden_layer_sizes=(100),
                                    activation='relu')))

# To evaluate each model in turn
results = []
names = []

print("\n============================================= ")
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10)
    cv_results = model_selection.cross_val_score(model, P_train, C_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
print("\n============================================= ")
   
## Making predictions on validation dataset (to avoid overfitting)

print("PREDICTIONS ON VALIDATION DATASET")
for name, model in models:
    model.fit(P_train, C_train)
    predictions = model.predict(P_test)
    print("\nModel:", name)
    print("Accuracy:", accuracy_score(C_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(C_test, predictions))
    print(classification_report(C_test, predictions))
    print("=============================================")

# =============================================================================
# (V) PREDICT BREAST CANCER USING GENERAL VALUES OF ARRAYS
# =============================================================================

clf = SVC()

clf.fit(P_train, C_train)
accuracy = clf.score(P_test, C_test)
print("Accuracy:", accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print("Prediction:", prediction, "\n")
