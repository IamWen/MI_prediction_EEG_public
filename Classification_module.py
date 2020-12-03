import sys
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, LeavePGroupsOut, LeaveOneGroupOut, GroupShuffleSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
import lightgbm as lgb
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

import ReadFiles as RF
import ExtractFeatures as EF


def preprocessing(feature_df):
    # Check for null
    null_df = pd.DataFrame(feature_df.isnull().mean(), columns=["null %"])
    null_cols = null_df[null_df["null %"] > 0]
    # View cleaned dataframe
    cleaned_feature_df = feature_df.drop(list(null_cols.index), axis=1)
    cleaned_feature_df.head()
    return cleaned_feature_df


def normalize_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    StandardScaler()
    mu = scaler.mean_
    std = scaler.scale_
    X_train2 = scaler.transform(X_train)
    X_test2 = scaler.transform(X_test)
    return X_train2, X_test2


def Logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("xgBoost accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))
    return


def SVM_classifier(X_train, X_test, y_train, y_test, best_params={'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}):
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    best_ker = best_params['kernel']

    clf = SVC(C=best_C, gamma=best_gamma, kernel=best_ker)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("SVM accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))

    # w0 = svm_model.intercept_
    # w0 = w0[..., np.newaxis]
    # weights = svm_model.coef_
    # aug_weights = np.concatenate((w0, weights), axis=1)
    # # print('weights: '+str(aug_weights.shape))
    # vec = svm_model.support_vectors_
    # print('SVM vec: '+str(vec))
    return


def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train, cv=5)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("Random forest accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()


def AdaBoost(X_train, X_test, y_train, y_test, best_params={'learning_rate': 1, 'n_estimators': 90}):
    best_n_estimators = best_params['n_estimators']
    best_learning_rate = best_params['learning_rate']

    clf = AdaBoostClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("xgBoost accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))
    return


def xgBoost(X_train, X_test, y_train, y_test, best_params={'colsample_bytree': 0.5, 'eta': 0.05, 'gamma': 0.1, 'max_depth': 6, 'min_child_weight': 1}):
    best_eta = best_params['eta']
    best_max_depth = best_params['max_depth']
    best_min_child_weight = best_params['min_child_weight']
    best_gamma = best_params['gamma']
    best_col = best_params['colsample_bytree']

    clf = XGBClassifier(eta=best_eta, max_depth=best_max_depth,min_child_weight=best_min_child_weight,gamma=best_gamma,colsample_bytree=best_col)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("xgBoost accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))
    return


def lightGBM(X_train, X_test, y_train, y_test):
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("xgBoost accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))
    return


def cross_validation_grid_search(X, y, model, params):
    clf = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=-1, return_train_score=True)
    clf.fit(X, y)
    best_model = clf.best_estimator_
    best_params = clf.best_params_
    print("Best parameters: \n"+str(best_params))

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    print("Grid scores on development set:")
    for mean, std, param in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, param))
    results = clf.cv_results_
    return best_params


def classify_data(X_train, X_test, y_train, y_test):
    # random_forest(X_train, X_test, y_train, y_test)

    # model = AdaBoostClassifier()
    # params_AdaBoost = {"n_estimators":range(30, 101, 10), "learning_rate":[1, 0.1, 0.01]}
    # best_params = cross_validation_grid_search(X_train,y_train, model, params_AdaBoost)
    # AdaBoost(X_train, X_test, y_train, y_test, best_params)

    # model = SVC()
    # params_SVM = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}
    # best_params = cross_validation_grid_search(X_train,y_train, model, params_SVM)
    # SVM_classifier(X_train, X_test, y_train, y_test, best_params)

    # model = XGBClassifier()
    # params_xgBoost = {
    #     "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #     "min_child_weight": [1, 3, 5, 7],
    #     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    # }
    # best_params = cross_validation_grid_search(X_train, y_train, model, params_xgBoost)
    # xgBoost(X_train, X_test, y_train, y_test, best_params)

    model = lgb.LGBMClassifier()
    params_lightGBM = {"num_leaves":[26,31,36,41], "learning_rate":[0.05,0.1],"boosting_type":['gbdt','dart','goss']}
    best_params = cross_validation_grid_search(X_train, y_train, model, params_lightGBM)
    # lightGBM(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    dir = 'E:/USC/EE660_2020/data'
    eeg_epoch_full_df, W1, W2 = RF.read_epoched_data(dir)
    feature_df = EF.neuroDSP_alpha_instantaneous_amplitude_median(W1, eeg_epoch_full_df)
    cleaned_feature_df = preprocessing(feature_df)
    y = feature_df["y"]
    X = feature_df.drop("y",axis=1)
    X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = normalize_data(X_train0, X_test0)
    classify_data(X_train, X_test, y_train, y_test)
