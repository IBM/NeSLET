from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from imblearn.over_sampling import SMOTE
torch.manual_seed(0)
np.random.seed(0)
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)
# clf.predict([[2., 2.]])
#
# print(clf.support_vectors_)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",dest="data_path",type=str,help="Path to training data.")
parser.add_argument("--training_split_size",default=0.2,type=float,help="train split size")
parser.add_argument("--validation_split_size",default=0.1,type=float,help="train split size")
parser.add_argument("--classifier",default="svm",type=str,help="svm or logistic regression")

args = parser.parse_args()

print(args.data_path)
data = torch.load(args.data_path)
no_features  = data.shape[1] - 1
output_dir = 'NeSLET_everything/null_entity_classifier'

# random_permuation = list(np.random.permutation(len(data)))
# train_split_size = int(args.training_split_size*len(data))
# training_indexes = random_permuation[:train_split_size]
# val_split_size = int(args.validation_split_size*len(data))
# validation_indexes = random_permuation[train_split_size:train_split_size+val_split_size]
#
# train_data = data[training_indexes,:no_features]
# train_labels = data[training_indexes,-1]
#
# val_data = data[validation_indexes,:no_features]
# val_labels = data[validation_indexes,-1]
#
# test_indexes = random_permuation[train_split_size+val_split_size:]
#
# test_data = data[test_indexes,:no_features]
# test_labels = data[test_indexes,-1]
#
# # print(train_data.shape)
# # print(val_data.shape)
# # print(test_data.shape)
#
# train_data = train_data.tolist()
# train_labels = train_labels.detach().tolist()
# train_labels = [int(item) for item in train_labels]
#
# val_data = val_data.detach().tolist()
# val_labels = val_labels.detach().tolist()
# val_labels = [int(item) for item in val_labels]
#
#
# test_data = test_data.detach().tolist()
# test_labels = test_labels.detach().tolist()
# test_labels = [int(item) for item in test_labels]
#
#
# print("train data size {}x{}".format(len(train_data),len(train_data[0])))
# print("train labels size {}x1".format(len(train_labels)))
#
# print("val data size {}x{}".format(len(val_data),len(val_data[0])))
# print("val labels size {}x1".format(len(val_labels)))
#
# print("test data size {}x{}".format(len(test_data),len(test_data[0])))
# print("test labels size {}x1".format(len(test_labels)))
#
# clf = svm.SVC()
# clf.fit(train_data, train_labels)
# print(clf.support_vectors_)
X = data[:,:no_features]
Y =  data[:,-1]
X = X.tolist()
Y = Y.tolist()
Y_new = []
for label in Y:
    if label == 0:
        Y_new.append(1)
    else:
        Y_new.append(0)
Y = Y_new

# Split the dataset in two parts
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.7, random_state=0,stratify=Y)

print("Before OverSampling, counts of label '1': {}".format(sum(Y_train)))
print("Before OverSampling, counts of label '0': {} \n".format(len(Y_train)-sum(Y_train)))
#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# sm = SMOTE(random_state=1)
# X_train, Y_train = sm.fit_resample(X_train, Y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(Y_train)))
print("Before OverSampling, counts of label '0': {} \n".format(len(Y_train)-sum(Y_train)))

cv = StratifiedKFold(n_splits=5)

#svc = svm.LinearSVC(random_state=0, tol=1e-5,max_iter=1000)
if args.classifier == 'svm':
    classifier_model = svm.SVC(random_state=0)
    # Set the parameters by cross-validation
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100], 'class_weight': [{0: w} for w in [0.1,1, 2, 4, 8]]}]
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='micro'),
               'recall': make_scorer(recall_score, average='micro'),
               'f1': make_scorer(f1_score, average='binary')
               }
else:
    classifier_model = LogisticRegression(random_state=0)
    # Set the parameters by cross-validation
    tuned_parameters = [
        {'C': [0.1, 1, 10, 100], 'class_weight': [{1: w} for w in [0.1, 1, 2, 4, 8]]}]
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='micro'),
               'recall': make_scorer(recall_score, average='micro'),
               'f1': make_scorer(f1_score, average='micro')}
    args.classifier = 'logistic_regression'
clf = GridSearchCV(classifier_model, tuned_parameters,verbose=3,n_jobs=8,cv=cv,scoring = scoring, refit='f1')
clf.fit(X_train, Y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
params = clf.get_params(deep=True)
print(params)
best_model = clf.best_estimator_
print('w = ',best_model.coef_)
print('b = ',best_model.intercept_)

if args.classifier == 'svm':
    print('Indices of support vectors = ', best_model.support_)
    print('Support vectors = ', best_model.support_vectors_)
    print('Number of support vectors for each class = ', best_model.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(best_model.dual_coef_))

Y_true, Y_pred = Y_test, clf.predict(X_test)
print(classification_report(Y_true, Y_pred))
print(clf.decision_function(X_test))
print(Y_pred)

output_dir = os.path.join(output_dir,args.classifier)
# save the model to disk
filename = os.path.join(output_dir,'final_model.pkl')
pickle.dump(clf, open(filename, 'wb'))