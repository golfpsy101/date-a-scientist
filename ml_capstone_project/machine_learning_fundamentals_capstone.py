#!/usr/bin/env python
# coding: utf-8

"""
This module attempts to determine if an individual's age, body type, diet,
job and education can predict their gender.
"""

import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split


# Create a dataframe.
df = pd.read_csv("profiles.csv")

print(df.describe())

# Columns of interest.
columns = ['age', 'body_type', 'diet', 'job', 'education', 'sex']
# Create a dataframe with just the columns of interest.
profiles = df[columns].copy()

# Explore the data
print(profiles['age'].head())
print(profiles['age'].value_counts()[:5])
print(profiles['age'].value_counts()[-5:])
print(profiles['body_type'].head())
print(profiles.body_type.value_counts())

# Drop the NaNs, ages over 79, and 'rather not say' body types.
profiles = profiles.dropna()
profiles = profiles.query('age < 80')
profiles = profiles.query('body_type != ["rather not say"]')

# Create some plots.
fig = plt.figure()
plt.tight_layout()
plt.title("Body Type vs Diet")
plt.scatter(profiles.body_type, profiles.diet, alpha=0.2)
plt.xticks(rotation=45)
plt.savefig("body_type_vs_diet.png")
plt.show()

plt.title("Age vs Body Type")
plt.bar(profiles.age, profiles.body_type, alpha=0.2)
plt.savefig("age_vs_body_type.png")
plt.show()

# Convert the 'sex' column to a numerical value: {Female: 0, Male: 1}.
profiles['sex_code'] = profiles['sex'].astype("category").cat.codes
y = profiles.sex_code
y = y.ravel() # change it from a 1D array to a 2D array.

print(profiles[['sex', 'sex_code']].head())
print(profiles[['sex', 'sex_code']].tail())

# Convert category data to numerical data.
rev_cols = ['age', 'body_type', 'diet', 'job', 'education']
# Create a copy of the dataframe.
profiles_ohe = profiles[rev_cols].copy()
# Transform the data.
X = pd.get_dummies(profiles_ohe,
                   columns=rev_cols,
                   prefix=rev_cols
                   )

# Split the dataframe.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
                                                    )

# Concatenate the 'body_type' values and some of the encoded values.
body_types = pd.concat([profiles['body_type'],
                        X['body_type_a little extra'],
                        X['body_type_average'],
                        X['body_type_thin']],
                        axis=1,
                        join_axes=[profiles.index]
                       )

print(body_types.head())

# Start the overall timer.
start_time = time.time()

# Prepare configuration for cross validation testing.
seed = 7

# Prepare models.
models = []
models.append(("CART", DecisionTreeClassifier(class_weight='balanced')))
models.append(("KNN", KNeighborsClassifier(weights='distance')))
models.append(("LR", LogisticRegression(solver='liblinear',
                                        multi_class='auto',
                                        max_iter=1000,
                                        class_weight='balanced'
                                        )
               )
              )
models.append(("SGD", SGDClassifier(max_iter=1000,
                                    tol=1e-3,
                                    early_stopping=True
                                    )
               )
              )
models.append(("GNB", GaussianNB()))
models.append(("MNB", MultinomialNB()))

# Evalute each model.
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    model_start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
            model, X, y, cv=kfold, scoring=scoring
            )

    results.append(cv_results)
    names.append(name)
    print("[{0}] CV Mean: {1:0.5f}, Std: {2:0.5f}".format(
            name, cv_results.mean(),cv_results.std()
            )
          )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("[{0}] Train Score: {1:0.5f}".format(
            name, model.score(X_train, y_train)
            )
          )

    print("[{0}] Test Score: {1:0.5f}".format(
            name, model.score(X_test, y_test)
            )
          )

    print("[{0}] Metrics Accuracy: {1:0.5f}".format(
            name, metrics.accuracy_score(y_test, y_pred)
            )
          )

    print("[{0}] Metrics Report: \n{1}".format(
            name, metrics.classification_report(
                    y_test, y_pred,
                    target_names=['female', 'male']
                    )
            )
          )

    # Print the time each model took to run.
    print("{0} Model: Elapsed Time (s): {1:0.3f}\n\n".format(
            name, time.time() - model_start_time)
          )

print("Total Elapsed Time: {0:0.3f} seconds.".format(
        time.time() - start_time)
      )

# Plot the algorithm comparisons results.
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig('algorithm_comparison_results.png')
plt.show()

