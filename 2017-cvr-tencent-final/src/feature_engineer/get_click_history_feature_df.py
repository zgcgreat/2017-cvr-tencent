# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
from datetime import datetime
from collections import Counter

begin = datetime.now()


def get_history_label(day, dfTrain, dfTest, dfApp_action, dfApp_installed):
    click_end = day * 10000
    click_begin = (day - 4) * 10000
    if day != 31:
        dfVal = dfTrain.ix[
            (dfTrain['clickTime'] >= click_end) & (dfTrain['clickTime'] < click_end + 10000), ["user_app_values",
                                                                                               "label"]]
        dfTrain = dfTrain.ix[
            (dfTrain['clickTime'] >= click_begin) & (dfTrain['clickTime'] < click_end), ["user_app_values", "label"]]
        dfApp_action = dfApp_action.ix[
            (dfApp_action['installTime'] >= click_begin) & (dfApp_action['installTime'] < click_end), [
                "user_app_values", "label_a"]]
        dfApp_installed = dfApp_installed.ix[:, ["user_app_values", "label_i"]]
        dfVal = dfVal.ix[:, ["user_app_values"]]
        dfTest = dfVal
        dfTest = pd.merge(dfTest, dfTrain, on="user_app_values", how="left", sort=False)
        # 统计消极点击次数
        print(dfTest.values.shape)
        dfCount = dfTest.groupby("user_app_values").apply(lambda df: len(df[df["label"] == 0])).reset_index()
        dfCount.columns = ["user_app_values", "neg_click_count"]
        dfTest = pd.merge(dfTest, dfCount, on="user_app_values", how="left", sort=False)
        # 统计积极点击次数
        dfCount = dfTest.groupby("user_app_values").apply(lambda df: len(df[df["label"] == 1])).reset_index()
        dfCount.columns = ["user_app_values", "pos_click_count"]
        dfTest = pd.merge(dfTest, dfCount, on="user_app_values", how="left", sort=False)
        # 缺失值
        dfTest = dfTest.fillna({"label": 10, "neg_click_count": 0, "pos_click_count": 0})

        dfTest = pd.merge(dfTest, dfApp_action, on="user_app_values", how="left", sort=False)
        dfTest = dfTest.fillna(10)

        dfTest = pd.merge(dfTest, dfApp_installed, on="user_app_values", how="left", sort=False)
        dfTest = dfTest.fillna(10)

        print(dfTest.values.shape)
        print(dfTest.columns)
        dfTest["label_t"] = dfTest["label"].values
        test = dfTest["label_i"].values + dfTest["label_a"].values + dfTest["label_t"].values
        test[test == 20] = 31
        test[test < 30] = 1
        test[test == 30] = 2
        test[test == 31] = 0
        dfTest["history_label"] = test
        # 去除重复值，或者冲突值
        uav_max = dfTest.groupby("user_app_values").apply(lambda df: df["label_t"].values.max()).reset_index()
        uav_max.columns = ["user_app_values", "label_t_max"]
        dfTest = pd.merge(dfTest, uav_max, on="user_app_values", how="left", sort=False)
        print(dfTest.values.shape)
        dfTest = dfTest.drop_duplicates(["user_app_values"])
        dfTest = dfTest.ix[:,
                 ["user_app_values", "label_i", "label_a", "label_t_max", "history_label", "neg_click_count",
                  "pos_click_count"]]
        print(dfTest.values.shape)
        #        dfTest.to_csv("click_history_feature_val.csv",index=False)
        return dfTest
    else:
        dfTrain = dfTrain.ix[:, ["user_app_values", "label"]]
        dfTest = dfTest.ix[:, ["user_app_values"]]
        dfApp_action = dfApp_action.ix[:, ["user_app_values", "label_a"]]
        dfApp_installed = dfApp_installed.ix[:, ["user_app_values", "label_i"]]

        dfTest = pd.merge(dfTest, dfTrain, on="user_app_values", how="left", sort=False)
        # 统计消极点击次数
        print(dfTest.values.shape)
        dfCount = dfTest.groupby("user_app_values").apply(lambda df: len(df[df["label"] == 0])).reset_index()
        dfCount.columns = ["user_app_values", "neg_click_count"]
        dfTest = pd.merge(dfTest, dfCount, on="user_app_values", how="left", sort=False)
        # 统计积极点击次数
        dfCount = dfTest.groupby("user_app_values").apply(lambda df: len(df[df["label"] == 1])).reset_index()
        dfCount.columns = ["user_app_values", "pos_click_count"]
        dfTest = pd.merge(dfTest, dfCount, on="user_app_values", how="left", sort=False)
        # 缺失值
        dfTest = dfTest.fillna({"label": 10, "neg_click_count": 0, "pos_click_count": 0})

        dfTest = pd.merge(dfTest, dfApp_action, on="user_app_values", how="left", sort=False)
        dfTest = dfTest.fillna(10)

        dfTest = pd.merge(dfTest, dfApp_installed, on="user_app_values", how="left", sort=False)
        dfTest = dfTest.fillna(10)

        print(dfTest.values.shape)
        print(dfTest.columns)
        dfTest["label_t"] = dfTest["label"].values
        test = dfTest["label_i"].values + dfTest["label_a"].values + dfTest["label_t"].values
        test[test == 20] = 31
        test[test < 30] = 1
        test[test == 30] = 2
        test[test == 31] = 0
        dfTest["history_label"] = test
        print(dfTest.columns)
        # 去除重复值，或者冲突值
        uav_max = dfTest.groupby("user_app_values").apply(lambda df: df["label_t"].values.max()).reset_index()
        uav_max.columns = ["user_app_values", "label_t_max"]
        dfTest = pd.merge(dfTest, uav_max, on="user_app_values", how="left", sort=False)
        print(dfTest.values.shape)
        dfTest = dfTest.drop_duplicates(["user_app_values"])
        dfTest = dfTest.ix[:,
                 ["user_app_values", "label_i", "label_a", "label_t_max", "history_label", "neg_click_count",
                  "pos_click_count"]]
        print(dfTest.values.shape)

        return dfTest


# load data
data_root = "../../data"
dfTrain = pd.read_csv("%s/train.csv" % data_root)
dfVal = pd.read_csv("%s/daily_data/train_30.csv" % data_root)
dfTest = pd.read_csv("%s/test.csv" % data_root)
dfAd = pd.read_csv("../../data_ori/ad.csv")
dfApp_action = pd.read_csv("../../data_ori/user_app_actions.csv")
dfApp_installed = pd.read_csv("../../data_ori/user_installedapps.csv")
print(dfTest.values.shape)
dffeat = pd.read_csv("%s/single_double_feature_alldata.csv" % data_root)
# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID", sort=False)
dfVal = pd.merge(dfVal, dfAd, on="creativeID", sort=False)
dfTest = pd.merge(dfTest, dfAd, on="creativeID", sort=False)
key1 = "userID"
key2 = "appID"
dfTrain["user_app_values"] = dfTrain[key1].values * 1e6 + dfTrain[key2].values
dfTest["user_app_values"] = dfTest[key1].values * 1e6 + dfTest[key2].values
dfVal["user_app_values"] = dfVal[key1].values * 1e6 + dfVal[key2].values
dfApp_action["user_app_values"] = dfApp_action[key1].values * 1e6 + dfApp_action[key2].values
dfApp_installed["user_app_values"] = dfApp_installed[key1].values * 1e6 + dfApp_installed[key2].values
dffeat["user_app_values"] = dffeat[
                                key1].values * 1e6 + dffeat[key2].values
# day<170000 to dfappinstalled
dfApp_installed = pd.concat([dfApp_installed, dfApp_action.ix[dfApp_action["installTime"] < 170000, :]], join="inner",
                            axis=0)
dfApp_action = dfApp_action.ix[dfApp_action["installTime"] >= 170000, :]
dfApp_action["label_a"] = 1
dfApp_installed["label_i"] = 1
dffeat_copy = dffeat.copy()
dffeat["label_i"] = 0
dffeat["label_a"] = 0
dffeat["label_t_max"] = 0
dffeat["history_label"] = 0
dffeat["neg_click_count"] = 0
dffeat["pos_click_count"] = 0
# dfHis=get_history_label(31,dfTrain,dfTest,dfVal,dfApp_action,dfApp_installed)
for iter in range(21, 32):
    click_begin = iter * 10000
    click_end = (iter + 1) * 10000
    dfHis = get_history_label(iter, dfTrain, dfTest, dfApp_action, dfApp_installed)

    dfTemp = pd.merge(
        dffeat_copy.ix[(dffeat_copy['clickTime'] >= click_begin) & (dffeat_copy['clickTime'] < click_end), :], dfHis,
        on="user_app_values", how="left", sort=False)
    dffeat.ix[
        (dffeat['clickTime'] >= click_begin) & (dffeat['clickTime'] < click_end), ["label_i", "label_a", "label_t_max",
                                                                                   "history_label", "neg_click_count",
                                                                                   "pos_click_count"]] = dfTemp.ix[:,
                                                                                                         ["label_i",
                                                                                                          "label_a",
                                                                                                          "label_t_max",
                                                                                                          "history_label",
                                                                                                          "neg_click_count",
                                                                                                          "pos_click_count"]].values
    #    print dffeat[dffeat["history_label"]==1]
    #    print dfHis[dfHis["history_label"]==1].values.shape
    #    print dfTemp.values.shape
    #    print dfTemp[dfTemp["history_label"]==1].values.shape
    #    print dffeat[dffeat["history_label"]==1].values.shape
    print(iter)
dffeat.to_csv("single_double_feature_alldata_history.csv", index=False)
print(datetime.now() - begin)
