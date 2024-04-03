import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt
from sklearn import datasets, ensemble, model_selection
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

# transforming csv into a DataFrame object
# import data and split
df_lag = pd.read_csv("rawdata.csv")
train_set = df_lag.iloc[:,0:48]
labels = df_lag.iloc[:,48:49]
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_set, labels, random_state=0)

# model and fit
cls_t = tree.DecisionTreeClassifier()
cls_t.fit(X_train, y_train);

# the feature importance rank
#print(cls_t.feature_importances_)

# -- feature importance graph
#importances = cls_t.feature_importances_
#indices = np.argsort(importances)
#features = df_lag.columns
#plt.title('Feature Importances')
#j = 11   # top j importance
#plt.barh(range(j), importances[indices][len(indices)-j:], color='g', align='center')
#plt.yticks(range(j), [features[i] for i in indices[len(indices)-j:]])
#plt.xlabel('Relative Importance')
#plt.show()

# first visualization
fig = plt.figure(figsize=(16, 8))
vis = tree.plot_tree(cls_t, feature_names = df_lag.columns.to_list(), class_names = ['Not stressed', 'Middle Value', 'Stressed'], max_depth=3, fontsize=9, proportion=True, filled=True, rounded=True)
plt.savefig('tree4.eps',format='eps',bbox_inches = "tight")


#clf = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
#clf.fit(X_train, y_train)
#predictions = clf.predict(X_train)
#cls_t = tree.DecisionTreeClassifier()
#cls_t.fit(X_train, predictions)

#fig = plt.figure(figsize=(16, 8))
#vis = tree.plot_tree(cls_t, feature_names = df_lag.columns.to_list(), class_names = ['Not stressed', 'Middle Value', 'Stressed'], max_depth=3, fontsize=9, proportion=True, filled=True, rounded=True)
#plt.savefig('tree4.eps',format='eps',bbox_inches = "tight")

