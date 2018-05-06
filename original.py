import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('skill.csv')
dataset=dataset[dataset['TotalHours']<5000]
X = dataset.iloc[:, 1:19].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

######################## all the models ############################
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

names = [
"Perceptron",
"LogisticRegression",
"SVC",
"Decision Tree",
"Random Forest",
"RBF SVM",
"Neural Net",
"Naive Bayes",
"Nearest Neighbors"
]

classifiers = [
Perceptron(),
LogisticRegression(),
SVC(C=100),
DecisionTreeClassifier(max_depth=5),
RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
SVC(kernel='rbf',gamma=2, C=1),
MLPClassifier(hidden_layer_sizes=(100,50), alpha=1),
GaussianNB(),
KNeighborsClassifier(5)
]
score1=[]

for name, clf in zip(names, classifiers):
pipe = Pipeline([('clf', clf)])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print("Algo: {0:<20s} and Score: {1:0.4f}".format(name, score))
score1.append(score)
print(' ')
print('{0} performed the best with accuracy {1:0.4f}'.format (names [np.argmax(score1)],np.amax(score1)))

################### Add PCA ####################################
from sklearn.decomposition import PCA
print ('')
print('Accuracy after adding PCA')
for name, clf in zip(names, classifiers):
pipe = Pipeline( [('pca', PCA(n_components = 4)),
('clf', clf)])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print("Algo: {0:<20s} and Score: {1:0.4f}".format(name, score))

################### 10-fold cross_validation #######################
from sklearn.model_selection import cross_val_score
for name, clf in zip(names, classifiers):
scores = cross_val_score(clf, X, y, cv=10)
print("Algo: {0:<20s} and Score: {1:0.4f}".format(name, np.mean(scores)))



################## confusion_matrix ##############################
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
for name, clf in zip(names, classifiers):
pipe = Pipeline( [('pca', PCA(n_components = 5)),
('clf', clf)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
for j in range(confmat.shape[1]):
ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

print(normalize(confmat, axis=1, norm='l1'))
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
print(name)
print(clf)

# Employ GridSearchCV() to find the optimal parameters for SVC's kernel = 'rbf',
from sklearn.model_selection import GridSearchCV

clf = SVC(kernel="rbf")
c = [0.01,1,10,100]
gamma = [0.001, 0.01,0.1, 1.0,100.0]

pipe = Pipeline( [ ('scl', StandardScaler()),
('clf', SVC(C=100) ) ] )

pipe.fit(X_train,y_train) # <- note that pipe line inherits clf methods
print(pipe.score(X_test,y_test)) # notably .fit(), .predict() and .score()

params = {'clf__kernel': ['rbf'],
'clf__C' : c,
'clf__gamma' : gamma}

grid = GridSearchCV(estimator = pipe,
param_grid = params,
cv = 3)

grid.fit(X_train, y_train)
print(grid.best_estimator_)
print(' ')
print(grid.best_score_)
print(' ')
print(grid.score(X_test, y_test))
print(' ')
print(grid.best_params_)

######## nested cross-validation csv ############
gs = GridSearchCV(estimator=pipe,
param_grid=params,
scoring='accuracy',
cv=2)

scores = cross_val_score(gs, X_train, y_train,
scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
np.std(scores)))


# Employ GridSearchCV() to find the optimal parameters for DecisionTreeClassifier
clf = DecisionTreeClassifier()
c = [5,10,15,20,25,30]


pipe = Pipeline( [ ('scl', StandardScaler()),
('clf', DecisionTreeClassifier(max_depth=5) ) ] )

params = {'clf__max_depth' : c }
score_types=['accuracy','roc_auc','f1']
print('Grid search for DecisionTree')
print(' ')
for scoret in score_types:
grid = GridSearchCV(estimator = pipe,
param_grid = params,
scoring=scoret,
cv = 3)
grid.fit(X_train, y_train)
print("Score based on", scoret)
print('best training score',grid.best_score_)
print('best testing score',grid.score(X_test, y_test))
print('best params',grid.best_params_)
print(' ')
######## nested cross-validation DecisionTree ############
gs = GridSearchCV(estimator=pipe,
param_grid=params,
scoring='accuracy',
cv=2)

scores = cross_val_score(gs, X_train, y_train,
scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
np.std(scores)))


# Employ GridSearchCV() to find the optimal parameters for RandomForestClassifier
clf = RandomForestClassifier()
c = [5,7,9,11]
n_estimators = [90,95,100,105]
max_features = [1,3,5,7]

pipe = Pipeline( [ ('scl', StandardScaler()),
('clf', RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1) ) ] )

params = {'clf__max_depth' : c,
'clf__n_estimators' : n_estimators,
'clf__max_features' : max_features
}
score_types=['accuracy','roc_auc','f1']
for scoret in score_types:
grid = GridSearchCV(estimator = pipe,
param_grid = params,
scoring=scoret,
cv = 3)

grid.fit(X_train, y_train)
print("Score based on", scoret)
print('best training score',grid.best_score_)
print('best testing score',grid.score(X_test, y_test))
print('best params',grid.best_params_)
print(' ')
######## nested cross-validation RandomForest ############
gs = GridSearchCV(estimator=pipe,
param_grid=params,
scoring='accuracy',
cv=2)

scores = cross_val_score(gs, X_train, y_train,
scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
np.std(scores)))

# Employ GridSearchCV() to find the optimal parameters for KNN
clf = KNeighborsClassifier()
n_neighbors = [5,8,10,12,15]

pipe = Pipeline( [ ('scl', StandardScaler()),
('clf', KNeighborsClassifier(5) ) ] )

params = {'clf__n_neighbors' : n_neighbors,
}
score_types=['accuracy','roc_auc','f1']
print('Grid search for knn')
print(' ')
for scoret in score_types:
grid = GridSearchCV(estimator = pipe,
param_grid = params,
scoring=scoret,
cv = 3)

grid.fit(X_train, y_train)
print("Score based on", scoret)
print('best training score',grid.best_score_)
print('best testing score',grid.score(X_test, y_test))
print('best params',grid.best_params_)
print(' ')

########## nested cross-validation KNN ############
gs = GridSearchCV(estimator=pipe,
param_grid=params,
scoring='accuracy',
cv=2)

scores = cross_val_score(gs, X_train, y_train,
scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
np.std(scores)))


############## leanring curve ###################
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
for name, clf in zip(names, classifiers):
pipe = Pipeline( [('pca', PCA(n_components = 5)),
('clf', clf)])
train_sizes, train_scores, test_scores =\
learning_curve(estimator=pipe,
X=X_train,
y=y_train,
train_sizes=np.linspace(0.1, 1.0, 10, 100),
cv=10,
n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
color='blue', marker='o',
markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
train_mean + train_std,
train_mean - train_std,
alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
color='green', linestyle='--',
marker='s', markersize=5,
label='validation accuracy')

plt.fill_between(train_sizes,
test_mean + test_std,
test_mean - test_std,
alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 0.9])
plt.tight_layout()
plt.title('Learning Curve')
plt.show()