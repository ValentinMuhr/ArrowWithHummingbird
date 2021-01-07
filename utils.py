import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
#%matplotlib inline
sns.set_style("whitegrid")
sns.set(rc = {'figure.figsize':(20, 15)})

import sklearn
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from matplotlib import pyplot as plt
import time
import pickle

from rfpimp import permutation_importances
import eli5
from eli5.sklearn import PermutationImportance
from multiprocessing import Manager, Pool


def split(df, label):
    y = df["quality"]
    X = df.drop(["quality"], inplace=False, axis=1)

    X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=.3, random_state=42)

    # combine only test data
    baseline_test = pd.concat([X_test,y_test], axis=1)

    #add quality as column name
    sym = [x for x in X_test.columns]
    sym.append("quality")
    baseline_test.columns = sym

    #get only good cells from test set
    baseline_test = baseline_test[baseline_test["quality"] == 1]

    # combine good test cells with original flagged cells
    clean_test = pd.concat([baseline_test])

    y_clean = clean_test["quality"]
    X_clean = clean_test.drop(["quality"], inplace=False, axis=1)
    print("for training:\n    X_train_"+label+", y_train_"+label+"\n\nfor testing:\n    X_test_"+label+", y_test_"+label)
    print("")
    return X_train, X_test, y_train, y_test



def tree_grd_sr(X_train, Y_train, f_name):
    ## Tuning the decision tree parameters and implementing cross-validation using Grid Search
    print("Starting Grid Search for Decision Tree...")
    dtc = tree.DecisionTreeClassifier(random_state=42)
    grid_param = {'criterion': ['entropy', 'gini'],
                  'splitter': ['best', 'random'],
		  'min_samples_split': [2, 3, 5, 7, 9],
                  'max_features': [None, 'auto', 'log2'],
                  'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]}
    # 'ccp_alpha':[0.0,0.001,0.01,0.1,1.0]}

    gd_sr = GridSearchCV(estimator=dtc, param_grid=grid_param, scoring='f1', cv=5, verbose=1, n_jobs=-1)

    # """
    # In the above GridSearchCV(), scoring parameter should be set as follows:
    # scoring = 'accuracy' when you want to maximize prediction accuracy
    # scoring = 'recall' when you want to minimize false negatives
    # scoring = 'precision' when you want to minimize false positives
    # scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
    # """

    gd_sr.fit(X_train, Y_train)
    best_params = gd_sr.best_params_

    print(best_params, "\n")
    np.save(f_name + 'best_params.npy', best_params)

    best_result = gd_sr.best_score_  # Mean cross-validated score of the best_estimator
    print(best_result)
    return best_params

def decision_tree(X_train, y_train, X_test, y_test, best_params):
    #print("Train Decision Tree ...")
    start = time.time()
    # for best recall set (entropy, min_samples_split = 2, splitter=random)
    # for best precision set (entropy, min_samples_split = 3)
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=3, random_state=41)
    if best_params != False:
        clf.set_params(**best_params)
    clf.fit(X_train, y_train)

    end_train = time.time()
    print("Learning duration:", time.time() - start)

    predictions = clf.predict(X_test)
    dt = accuracy_score(y_test, predictions)

    print("Predicting duration:", time.time() - end_train)
    print("Accuracy", dt)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return clf

def rf_grd_sr(X_train, Y_train, f_name):
    ## Tuning the decision tree parameters and implementing cross-validation using Grid Search
    print("Starting Grid Search for Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    grid_param = {'n_estimators': [100, 200, 300, 500, 800],
                  'criterion': ['entropy','gini'],
                  'min_samples_split':[2,3,5,7],
                  'max_features': [None, 'auto', 'log2'],
                  'min_impurity_decrease': [0.0, 0.01]}

    gd_sr = GridSearchCV(estimator=rf, param_grid=grid_param, scoring='recall', cv=5, verbose=1, n_jobs=-1)

    # """
    # In the above GridSearchCV(), scoring parameter should be set as follows:
    # scoring = 'accuracy' when you want to maximize prediction accuracy
    # scoring = 'recall' when you want to minimize false negatives
    # scoring = 'precision' when you want to minimize false positives
    # scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
    # """

    gd_sr.fit(X_train, Y_train)
    best_params = gd_sr.best_params_

    print(best_params, "\n")
    np.save(f_name + 'best_params.npy', best_params)

    best_result = gd_sr.best_score_  # Mean cross-validated score of the best_estimator
    print(best_result)
    return best_params

def random_forest(X_train, y_train, jobs, best_params):
    start = time.time()
    #clf = RandomForestClassifier(random_state=42, n_jobs=jobs)
    if best_params != False:
        clf.set_params(**best_params)
    else:
        clf  = RandomForestClassifier(criterion="entropy", min_samples_split=2, random_state=42, n_jobs=jobs)
        #clf  = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    end_train = time.time()
    print("Learning duration:", time.time() - start)
    return clf


def test_clf(clf, name, X_clean, y_clean):
    print("Test:", name+ "\n")
    startTime = time.time()
    predictions = clf.predict(X_clean)
    print("Predicting duration:", time.time() - startTime)
    dt = accuracy_score(y_clean, predictions)
    print("Accuracy", dt)
    print("Confusion Matrix")
    print(confusion_matrix(y_clean, predictions))
    print("")
    print("Classification Report:")
    print(classification_report(y_clean, predictions))

def save_model(model, file_name, clf_dir, shape):
    pickle.dump(model, open(clf_dir+file_name+'.clf', 'wb'))


# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, filename, clf_dir):
    imp_df.columns = ['feature', 'feature_importance']
    sns.set_style("whitegrid")
    sns_plot = sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)
    #sns_ax = sns_plot.axes
    #sns_ax.set_ylim(-0.5,)
    #sns_plot.set(xlim=(-0.5, 1.1))
    plt.show()
    sns_plot.figure.savefig(clf_dir + filename,bbox_inches='tight')

def permutation_imp(clf,x_train, y_train):
    def r2(clf, x_train, y_train):
        return r2_score(y_train, clf.predict(x_train))

    perm_imp_rfpimp = permutation_importances(clf, x_train, y_train, r2)
    perm_imp_rfpimp.reset_index(drop=False, inplace=True)
    return perm_imp_rfpimp



def drop_col_feat_imp(model, X_train, y_train, random_state=42):
    # clone the model to have the exact same specification as the one initially trained
    #model_clone = clone(model)
    # set random_state for comparability
    #model_clone.random_state = random_state
    # training and scoring the benchmark model
    #model_clone.fit(X_train, y_train)
    benchmark_score = model.score(X_train, y_train)
    # list for storing feature importances
    #manager = Manager()
    importances = []

    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        print("computing model for drop feature: " + col)
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)
        
    importances_df = imp_df(X_train.columns, importances)
    return importances_df

def get_feature_imp(clf, X_train, y_train, model, clf_dir, case):
    #feature importance
    print("plotting default feature importance ...")
    base_imp = imp_df(X_train.columns, clf.feature_importances_)
    var_imp_plot(base_imp, 'Default feature importance', model + case + "_dffimp.png", clf_dir)
    base_imp.to_csv(clf_dir + model + case + "_dffimp.csv", index=False)

    #permutation importance
    print("plotting permutation feature importance ...")
    perm_imp = permutation_imp(clf, X_train, y_train)
    var_imp_plot(perm_imp, 'Permutation feature importance', model + case + "_permfimp.png", clf_dir)
    perm_imp.to_csv(clf_dir + model + case + "_permfimp.csv", index=False)

    #drop columns importance
    print("plotting drop columns feature importance ...")
    drop_imp = drop_col_feat_imp(clf, X_train, y_train)
    var_imp_plot(drop_imp, 'Drop Column feature importance', model + case + "_dropfimp.png", clf_dir)
    drop_imp.to_csv(clf_dir + model + case + "_dropfimp.csv", index=False)

#if __name__ == "__main__":
#    pass
