'''The following code for feature extraction was modified from https://github.com/NeuroTech-UCSD/Neural-Data-Analysis-Notebooks'''

import sys
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization

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
    return indices


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


def xgBoost(X_train, X_test, y_train, y_test, best_params={'colsample_bytree': 0.4, 'eta': 0.2, 'gamma': 0.0, 'max_depth': 12, 'min_child_weight': 1}):
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


def lightGBM(X_train, X_test, y_train, y_test, best_params={'boosting_type': 'dart', 'learning_rate': 0.1, 'num_leaves': 41}):
    best_boost = best_params['boosting_type']
    best_learning_rate = best_params['learning_rate']
    best_num_leaves = best_params['num_leaves']
    clf = lgb.LGBMClassifier(boosting_type=best_boost,learning_rate=best_learning_rate, num_leaves=best_num_leaves)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("lightGBM accuracy: "+ str(test_acc1))
    target_names = ['class 1','class 2']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))
    return


def CNN(eeg_epoch_full_df):
    train_df = eeg_epoch_full_df
    train_df.head()

    # Stack arrays to form 2d matrices for each trial
    X = train_df.drop(["patient_id", "start_time", "event_type"], axis=1).apply(lambda x: np.stack(x, axis=-1), axis=1)
    X = np.array(X.values.tolist())
    X = X.reshape(list(X.shape) + [1])
    # Convert labels to numpy array
    Y = train_df["event_type"].values.astype(float)

    X_tot, X_test, y_tot, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tot, y_tot, test_size=0.2, random_state=0)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 6, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train,
              epochs=30,
              batch_size=64,
              validation_data=(X_val, y_val))
    test_loss, test_acc = model.evaluate(X_val, y_val, batch_size=64)
    print(test_acc)
    y_pred = model.predict_classes(X_test)
    test_acc1 = metrics.accuracy_score(y_test, y_pred)
    print("Test accuracy: "+ str(test_acc1))
    report = metrics.classification_report(y_test, y_pred, labels=[0, 1])
    print(report)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n'+str(conf_matrix))

    print(history.history.keys())
    # summarize history for accuracy
    fig1,ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    fig2,ax2 = plt.subplots()
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    return


def cross_validation_grid_search(X, y, model, params):
    clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1, return_train_score=True)
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
    return best_params, results


def plot_cv_results(results, param):
    means = results['mean_test_score']
    stds = results['std_test_score']

    plt.figure(figsize=(13, 13))
    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[param].data, dtype=float)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.legend(loc="best")
    plt.grid(False)
    plt.xlabel(param +"(log scale)")
    plt.ylabel("Accuracy")
    plt.title("GridSearchCV evaluating using accuracy",fontsize=16)
    ax.errorbar(X_axis, means, stds, linestyle='None', fmt='-^', capsize=4,capthick=2)


def select_features(X_tot, y_tot):
    X_train, X_val, y_train, y_val = train_test_split(X_tot, y_tot, test_size=0.2, random_state=0)
    ind = random_forest(X_train, X_val, y_train, y_val)

    X_train = X_train[:, ind[:80]]
    X_val = X_val[:, ind[:80]]
    print(X_tot.shape)
    print(X_train.shape)
    random_forest(X_train, X_val, y_train, y_val)
    return ind


def select_model(X_train, y_train):
    # model = AdaBoostClassifier()
    # params_AdaBoost = {"n_estimators":range(30, 101, 10), "learning_rate":[1, 0.1, 0.01], "algorithm": ['SAMME', 'SAMME.R']}
    # best_params, results = cross_validation_grid_search(X_train,y_train, model, params_AdaBoost)
    # # 0.697(+ / -0.028) for {'algorithm': 'SAMME', 'learning_rate': 1, 'n_estimators': 100}

    # model = SVC()
    # params_SVM = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf', 'sigmoid']}
    # best_params, results = cross_validation_grid_search(X_train,y_train, model, params_SVM)
    # # 0.732(+ / -0.042) for {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}

    # model = XGBClassifier()
    # params_xgBoost = {
    #     "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #     "min_child_weight": [1, 3, 5, 7],
    #     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    # }
    # best_params, results = cross_validation_grid_search(X_train, y_train, model, params_xgBoost)
    # # 0.743(+ / -0.015) for {'colsample_bytree': 0.4, 'eta': 0.2, 'gamma': 0.0, 'max_depth': 12, 'min_child_weight': 1}

    # model = lgb.LGBMClassifier()
    # params_lightGBM = {"num_leaves":[26,31,36,41], "learning_rate":[0.02, 0.05, 0.1],"boosting_type":['gbdt','dart','goss']}
    # best_params, results = cross_validation_grid_search(X_train, y_train, model, params_lightGBM)
    # # 0.727(+ / -0.029) for {'boosting_type': 'dart', 'learning_rate': 0.1, 'num_leaves': 41}
    return best_params


def classify_data(X_train, X_test, y_train, y_test):
    xgBoost(X_train, X_test, y_train, y_test)
    # lightGBM(X_train, X_test, y_train, y_test)
    # SVM_classifier(X_train, X_test, y_train, y_test)
    # AdaBoost(X_train, X_test, y_train, y_test)
    return


if __name__ == "__main__":
    dir = 'E:/USC/EE660_2020/data'
    eeg_epoch_full_df, _, _ = RF.read_epoched_data(dir)
    # feature_df = EF.get_all_features(eeg_epoch_full_df)
    # feature_df.to_pickle(dir+"/Wen_feature_df.pkl")
    feature_df = RF.read_my_features(dir)
    print(list(feature_df.columns))

    cleaned_feature_df = preprocessing(feature_df)
    y = feature_df["y"]
    X = feature_df.drop("y",axis=1)
    X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_test = normalize_data(X_train0, X_test0)

    ind = select_features(X_train, y_train)
    X_train = X_train[:, ind[:100]]
    X_test = X_test[:, ind[:100]]

    best_params = select_model(X_train, y_train)

    classify_data(X_train, X_test, y_train, y_test)

    # CNN(eeg_epoch_full_df)
    # plt.show()