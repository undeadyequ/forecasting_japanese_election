import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import logging
import argparse
import time
import json


pd.options.display.float_format = '{:.2f}'.format

def random_select_data(data, s, data_condition=("N2012")):
    csvdata = pd.read_csv(data, delimiter=",")

    if "U2009" in data_condition:
        csvdata = csvdata[csvdata['Year'] < 2010]
    if "N2012" in data_condition :
        csvdata = csvdata[csvdata['Year'] != 2012]
    if "N2009" in data_condition:
        csvdata = csvdata[csvdata['Year'] != 2009]

    np.random.seed(seed=s)
    csvdata = csvdata.reindex(np.random.permutation(csvdata.index))

    # 学習データとテストデータへの分割
    data_train = csvdata.iloc[int(len(csvdata) / 4):, :]
    data_test = csvdata.iloc[:int(len(csvdata) / 4), :]
    x_train = data_train.drop(['Year', 'LDP_votes', 'LDP_seats', 'PM_approval2'], axis=1)
    y_train = data_train['LDP_seats']
    x_test = data_test.drop(['Year', 'LDP_votes', 'LDP_seats', 'PM_approval2'], axis=1)
    y_test = data_test['LDP_seats']
    return x_train, y_train, x_test, y_test


def train_clf(method, seeds_num, title, csv_data, data_condition):
    train_maes = []
    test_maes = []
    train_rmses = []
    test_rmses = []


    feature_importances = []
    r_sqrs = []
    estimators = []

    y_preds = []
    y_preds_index = []

    x_tests = []
    y_tests = []

    for s in range(2, seeds_num+2):
        x_train, y_train, x_test, y_test = random_select_data(csv_data, s, data_condition)
        cvnum = x_train.shape[0]
        myscoring = "neg_mean_absolute_error"

        if method == "lr":
            besthyper = {'fit_intercept': True, 'normalize': False}
            clf = LinearRegression(**besthyper)

        elif method == "dt":
            besthyper = {'splitter': 'random', 'random_state': 0, 'max_depth': 3, 'criterion': 'mse'}
            clf = DecisionTreeRegressor(**besthyper)

        elif method == "svr":
            best_hyper = {'kernel': 'poly'}
            clf = SVR(**best_hyper)

        elif method == "rf":
            best_hyper = {
                'criterion': 'mae',
                'max_depth': 4,
                'max_features': 'auto',
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 125,
                'n_jobs': 1,
                'random_state': 0}
            clf = RandomForestRegressor(**best_hyper)

        elif method == "bag_lr":
            best_hyper = {
                'base_estimator': LinearRegression(),
                'bootstrap': True,
                'max_features': 3,
                'max_samples': 12,
                'n_estimators': 2000,
                'n_jobs': 1,
                'random_state': 0}
            clf = BaggingRegressor(**best_hyper)

        elif method == "bag_dt":
            best_hyper = {
                'base_estimator': DecisionTreeRegressor(),
                'bootstrap': True,
                'max_features': 3,
                'max_samples': 12,
                'n_estimators': 2000,
                'n_jobs': 1,
                'random_state': 0}
            clf = BaggingRegressor(**best_hyper)

        elif method == "adaboost_lr":
            besthyper = {
                'base_estimator': LinearRegression(),
                'learning_rate': 1.75,
                'loss': 'linear',
                'n_estimators': 10,
                'random_state': 0
            }
            clf = AdaBoostRegressor(**besthyper)

        elif method == "adaboost_dt":
            besthyper = {
                'base_estimator': DecisionTreeRegressor(max_depth=4),
                'learning_rate': 1.75,
                'loss': 'linear',
                'n_estimators': 10,
                'random_state': 0
            }
            clf = AdaBoostRegressor(**besthyper)

        elif method == "gradient_boost":
            besthyper = {'random_state': 0,
                         'loss': "ls",
                         'n_estimators': 40,
                         'min_samples_split': 3,
                         'min_samples_leaf': 2,
                         'max_features': 'auto',
                         'learning_rate': 0.3,
                         'max_depth': 5,
                         'criterion': 'friedman_mse'}
            besthyper = {
                'criterion': 'mae',
                'learning_rate': 0.2,
                'loss': 'lad',
                'max_depth': 3,
                'max_features': 'auto',
                'min_samples_leaf': 3,
                'min_samples_split': 2,
                'n_estimators': 125,
                'random_state': 0} # shima
            clf = GradientBoostingRegressor(**besthyper)

        elif method == "gradient_boost":
            besthyper = {
                'criterion': 'mae',
                'learning_rate': 0.2,
                'loss': 'lad',
                'max_depth': 3,
                'max_features': 'auto',
                'min_samples_leaf': 3,
                'min_samples_split': 2,
                'n_estimators': 125,
                'random_state': 0} # shima
            clf = GradientBoostingRegressor(**besthyper)

        elif method == "xgb":
            besthyper = {
                'booster': 'dart',
                'eta': 0.05,
                'eval_metric': 'mae',
                'gamma': 0,
                'lambda': 0.5,
                'max_depth': 5,
                'max_leaf_nodes': 1,
                'n_estimators': 100,
                'n_jobs': 1,
                'random_state': 0,
                'seed': 0,
                'verbosity': 0}
            clf = XGBRegressor(**besthyper)

        elif method == "xgb":
            besthyper = {
                'booster': 'dart',
                'eta': 0.05,
                'eval_metric': 'mae',
                'gamma': 0,
                'lambda': 0.5,
                'max_depth': 5,
                'max_leaf_nodes': 1,
                'n_estimators': 100,
                'n_jobs': 1,
                'random_state': 0,
                'seed': 0,
                'verbosity': 0}
            clf = XGBRegressor(**besthyper)

        else:
            clf = None
            print("Undefined clf")

        clf.fit(x_train, y_train)

        # Evaluation of train and test
        y_train_pred = clf.predict(x_train)
        y_train_pred_index = x_train.index
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        y_test_pred = clf.predict(x_test)
        y_test_pred_index = x_test.index
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Get prediction of LDP seats with index (both train and test)
        y_train_pred_cp = y_train_pred.copy()
        y_test_pred_cp = y_test_pred.copy()
        y_pred_index = y_train_pred_index.union(y_test_pred_index, sort=False)
        y_preds_index.append(y_pred_index)
        y_pred = np.hstack((y_train_pred_cp, y_test_pred_cp))

        # append result of different seeds to a list.
        train_maes.append(round(train_mae, 2))
        train_rmses.append(round(train_rmse, 2))
        test_maes.append(round(test_mae, 2))
        test_rmses.append(round(test_rmse, 2))
        y_preds.append(y_pred)

        # Evaluate by r_sqr only when method==lr
        estimators.append(clf)
        r_sqr = clf.score(x_test, y_test)
        r_sqrs.append(r_sqr)

        x_tests.append(x_test)
        y_tests.append(y_test)

    # Best hyper: find champion shuffle
    champion_number = np.argmin(test_maes)
    champ_estimator = estimators[champion_number]

    # find champion prediction
    y_pred_champ = [round(x) for x in y_preds[champion_number]]
    y_pred_index_y_pred_champ = y_preds_index[champion_number]
    y_pred_champ = pd.DataFrame(y_pred_champ, index=y_pred_index_y_pred_champ)

    # Got Permutation Feature Importance
    feats_important_dicts, feats_import_array = show_permutation_importance(champ_estimator, x_tests[champion_number], y_tests[champion_number])
    if method == "rf":
        feature_importance = champ_estimator.feature_importances_
        feature_importances.append(feature_importance)
        logging.info("ft importance: {}".format(feature_importances))

    #logging.info("average MAE of train(aver of {} shuffles): {}".format(seeds_num, np.mean(train_maes)))
    #logging.info("variacne MAE of train: {}".format(np.std(train_maes)))
    #logging.info("test result:{}".format(test_maes))
    #logging.info("average MAE of test (aver of {} shuffles): {}".format(seeds_num, np.mean(test_maes)))
    #logging.info("variacne MAE of test: {}".format(np.std(test_maes)))
    #logging.info("r_sqrs of test (aver of {} shuffles):: {}".format(seeds_num, r_sqrs))
    # Vis of Gap of pred and true
    #y_test_preds_champ[method] = y_test_preds[method][champion_number]
    #vis_y_pred_and_true(y_test_preds_champ, y_test)
    # lr: vis linear in 1-d, 2-d from feature importance
    return train_maes, train_rmses, test_maes, test_rmses, y_pred_champ, feats_important_dicts, feats_import_array


def vis_y_pred_and_true(y_preds, y_true):
    """
    :param y_preds: dict(method: y_pred)
    :param y_true:
    :return:
    """
    for method in y_preds.keys():
        x = np.arange(y_preds[method].shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("sample")
        ax.set_ylabel("ldp_seats")
        ax.plot(x, y_preds[method], label=method)
        ax.plot(x, y_true, label="gd")
    plt.legend()
    plt.savefig("pred_true_diff.png")


def show_permutation_importance(model, x_test, y_test):
    feat_import_dict = {}
    r = permutation_importance(model, x_test, y_test,
                               n_repeats=30,
                               random_state=0)
    logging.info("Feature Importance")

    for i in r.importances_mean.argsort()[::-1]:
        feat_import_dict[x_test.columns[i]] = [round(r.importances_mean[i], 3),
                                               round(r.importances_std[i], 3)]
        #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        #logging.info(f"{x_test.columns[i]:<8}"
        #      f"{r.importances_mean[i]:.3f}"
        #      f" +/- {r.importances_std[i]:.3f}")
    return feat_import_dict, r.importances


def eval_clf(method, clf):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all")
    parser.add_argument("--log", default="log_correct1.txt")
    parser.add_argument("--seed_num", type=int, default=100)
    parser.add_argument("--title", type=str, default="data_election_2020_correct2012")
    parser.add_argument("--resname", type=str, default="method_blr")

    csvdata = pd.read_csv("data_election_pred.csv", delimiter=",")

    args = parser.parse_args()
    logging.basicConfig(filename=args.log, filemode="a", format="%(asctime)s - %(message)s", level=logging.INFO)
    #csv_data = "data_election.csv"
    #csv_data = "data_election_2020.csv"
    #csv_data = "data_election_correct2012.csv"
    csv_data = "data_election_2020_correct2012.csv"

    res_dir = "exp"
    vis_subdir = "vis"

    #explore_data(csv_data)
    # explore_data_2d(csv_f)

    data_condition = ("1960-2017-N2012", "1960-2017-N2012-N2009", "1960-2021-N2012")
    # Condition: method, data
    # method:

    result = []
    methods = ["lr", "dt", "rf", "bag", "adaboost", "gradient_boost"]
    methods_wl = ["lr", "dt", "rf", "bag_lr", "bag_dt", "adaboost_lr", "adaboost_dt", "gradient_boost"]
    methods_xgb = ["lr", "dt", "rf", "bag", "adaboost", "xgb", "gradient_boost"]
    methods_blr = ["blr"]


    feats_arrays = {}
    for method in methods:
        #logging.info("***** Start Train by Method: {} with {} *****".format(method, csv_data))
        #logging.info("Config: {}".format(""))

        st = time.time()
        train_maes, train_rmses, test_maes, test_rmses, y_pred_champ, \
        feats_important_dicts, feats_import_array = \
            train_clf(method, args.seed_num, args.title, csv_data, data_condition=("N2012"))
        end = time.time()
        cls_res = [np.mean(train_maes), np.std(train_maes),
                   np.mean(train_rmses), np.std(train_rmses),
                   np.mean(test_maes), np.std(test_maes),
                   np.mean(test_rmses), np.std(test_rmses),
                   feats_important_dicts["DAYS"][0],
                   feats_important_dicts["DAYS"][1],
                   feats_important_dicts["GDP"][0],
                   feats_important_dicts["GDP"][1],
                   feats_important_dicts["PM_approval"][0],
                   feats_important_dicts["PM_approval"][1]
                   ]
        result.append(cls_res)
        print("{} time cost :{}".format(method, end-st))
        #logging.info("{} time cost :{}".format(method, (end-st)/3600))

        # add prediction result
        csvdata[method] = y_pred_champ
        feats_arrays[method] = feats_import_array.tolist()

    # accuracy csv
    with open("res_save.csv", "w") as f:
        for l in result:
            f.write(str(l) + "\n")

    with open("result/feature_import.json", "w") as f:
        json.dump(feats_arrays, f)

    pd_data = pd.DataFrame(np.array(result),
                           index=methods_wl,
                           columns=["train_mae", "train_mae_var",
                                    "train_rmse", "train_rmse_var",
                                    "test_mae", "test_mae_var",
                                    "test_rmse", "test_rmse_var",
                                    "imp_Days_mean", "imp_Days_std",
                                    "imp_GDP_mean", "imp_GDP_std",
                                    "imp_approval_mean", "imp_approval_std"
                                    ])
    print(pd_data)
    # pred_data and acc_data
    csvdata.to_csv(args.methods + "_pred.csv")
    pd_data.to_csv(args.methods + "_acc1.csv")