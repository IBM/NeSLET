import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

model_file_path = 'zel_everything/null_entity_classifier/svm/final_model.pkl'
#model_file_path = 'zel_everything/null_entity_classifier/logistic_regression//final_model.pkl'

data_path = 'zel_everything/ZEL/data_dinesh/aca/ZEL_unique_client_inferred_validated_1_fw_scores_added.t7'

data = torch.load(data_path)
no_features  = data.shape[1] - 1

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


# load the model from disk
loaded_model = pickle.load(open(model_file_path, 'rb'))
Y_pred = loaded_model.predict(X_test)


no_correct = 0
for index in range(len(Y_test)):
    if Y_test[index] == Y_pred[index]:
        no_correct += 1
print("Model Acc {}".format(no_correct/len(Y_test)))

print("Intial Acc if we assign majority label to all samples {}".format((len(Y_test)-sum(Y_test))/len(Y_test)))
print(classification_report(Y_test, Y_pred))

Y_pred_samples = []
for sample in X_test:
    item = np.expand_dims(sample, axis=0)
    y = loaded_model.predict(item)
    Y_pred_samples.append(y[0])

no_correct = 0
for index in range(len(Y_test)):
    if Y_test[index] == Y_pred_samples[index]:
        no_correct += 1
print(no_correct/len(Y_test))

print(classification_report(Y_test, Y_pred_samples))
# scores = loaded_model.decision_function(X_test)
# my_Y_pred = []
# for item in scores:
#     if item > 0:
#         my_Y_pred.append(1)
#     else:
#         my_Y_pred.append(0)
#
#
# no_correct = 0
# for index in range(len(Y_test)):
#     if Y_test[index] == my_Y_pred[index]:
#         no_correct += 1
# print(no_correct/len(Y_test))
#
#
# best_model = loaded_model.best_estimator_
#
# w = best_model.coef_
# b =  best_model.intercept_
#
# my_scores = np.dot(w,np.transpose(X_test)) + b
#
# # comparison = scores == my_scores[0]
# # equal_arrays = comparison.all()
# # print(equal_arrays)
#
#
# my_Y_pred_new = []
# for item in my_scores[0]:
#     if item > 0:
#         my_Y_pred_new.append(1)
#     else:
#         my_Y_pred_new.append(0)
#
#
# no_correct = 0
# for index in range(len(Y_test)):
#     if Y_test[index] == my_Y_pred_new[index]:
#         no_correct += 1
# print(no_correct/len(Y_test))