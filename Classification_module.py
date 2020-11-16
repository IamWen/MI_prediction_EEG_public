import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeavePGroupsOut, LeaveOneGroupOut, GroupShuffleSplit,cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def normalize_data(feature_train, feature_test):
    scaler = StandardScaler().fit(feature_train)
    StandardScaler()
    mu = scaler.mean_
    std = scaler.scale_
    feature_train2 = scaler.transform(feature_train)
    feature_test2 = scaler.transform(feature_test)
    return feature_train2, feature_test2, mu, std


def cross_validation(X, y, IDs):
    logo = LeaveOneGroupOut()

    for train_index, val_index in logo.split(X, y, groups=IDs.ravel()):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        SVM_classifier(X_train, X_val, y_train, y_val)
    return


def split_randomly(X, y):
    X_tot, X_test, y_tot, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tot, y_tot, test_size=0.2, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_by_ID(X, y, IDs):
    gss = GroupShuffleSplit(n_splits=5, train_size=.8, random_state=0)

    for train_index, test_index in gss.split(X, y, groups=IDs.ravel()):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test, _, _ = normalize_data(X_train, X_test)
        print('Training set shape: '+str(X_train.shape))
        print('Test set shape: '+str(X_test.shape))
        ID_tr, u_tr_ind = np.unique(IDs[train_index], return_index=True)
        ID_ts, u_ts_ind = np.unique(IDs[test_index], return_index=True)
        print('Training IDs: ' + str(ID_tr))
        print('Testing IDs: ' + str(ID_ts))

        # cross_validation(X_train,y_train,IDs[train_index])
        SVM_classifier(X_train, X_test, y_train, y_test)
    return


def SVM_classifier(X_train, X_test, y_train, y_test, best_C=1, best_gamma=0.001, best_ker='rbf'):
    svm_model = SVC(C=best_C, gamma=best_gamma, kernel=best_ker)
    svm_model.fit(X_train, y_train.ravel())
    y_pred = svm_model.predict(X_test)

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
    print("SVM accuracy: "+ str(test_acc1))

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


if __name__ == "__main__":
    dir = 'E:/USC/EE660_2020/data'
    X_list, y_list, n, labels = RF.read_raw_data(dir)