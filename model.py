import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_selection import RFE

# import data
years = [2018, 2019, 2020, 2021]
df = pd.DataFrame()
for year in years:
    a = pd.read_csv("texas_"+str(year)+"_season_stats_new.csv")
    a["season"] = year
    df = pd.concat([df, a])
df = df.reset_index().drop(["index"], axis=1)

# some data transformations
df["win"] = df["margin"].apply(lambda x: 1 if x > 0 else 0)

# one hot encoding for home court advantage
df["neutral"] = df["HomeCourt"].apply(lambda x: 1 if x == 0 else 0)
df["home"] = df["HomeCourt"].apply(lambda x: 1 if x == 1 else 0)
df["away"] = df["HomeCourt"].apply(lambda x: 1 if x == 2 else 0)

# one hot encoding for season
df["2018"] = df["season"].apply(lambda x: 1 if x == 2018 else 0)
df["2019"] = df["season"].apply(lambda x: 1 if x == 2019 else 0)
df["2020"] = df["season"].apply(lambda x: 1 if x == 2020 else 0)
df["2021"] = df["season"].apply(lambda x: 1 if x == 2021 else 0)

# create input and output data
df = df.drop(["HomeCourt", "margin", "season"], axis=1)

# train_data = df[df["2020"] == 0]
# test_data = df[df["2020"] != 0]
# xtrain = train_data.copy()
# ytrain = xtrain.pop("win")
# xtest = test_data.copy()
# print(len(xtest))
# ytest = xtest.pop("win")

x = df.copy()
# x = x.loc[:, ["top10","top15","top25","difffgp","diffftp","diffTovPerGame","neutral","away","win"]]
y = x.pop("win")

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2022, stratify=y)
# define GridSearch search space
search_space = [
    {
        "clf": [LogisticRegression()],
        "clf__C": [0.1, 1, 10, 20, 50, 100, 200, 300],
        "clf__class_weight": ["balanced", None],
        "clf__fit_intercept": [True, False],
        "clf__dual": [True, False],
        "clf__penalty": ["none", "l1", "l2", "elasticnet"]
    }
    # {
    #     "clf": [RandomForestClassifier()],
    #     "clf__n_estimators": [10, 20, 50, 100, 200],
    #     "clf__class_weight": ["balanced_subsample"],
    #     "clf__criterion": ["entropy", "gini"]
    # },
    # {
    #     "clf": [LinearRegression()],
    #     "clf__fit_intercept": [True, False],
    #     "clf__normalize": [True, False]
    # }
]

pipe = Pipeline([
    ("clf", None)
])

# fit estimator
gs = GridSearchCV(pipe, search_space, cv=3, scoring="f1_weighted", verbose=2, n_jobs=3)
# gs.fit(xtrain, ytrain)


clf = LogisticRegression(C=10, class_weight='balanced')
clf.fit(xtrain, ytrain)

ypred_probs = clf.predict_proba(xtest)
ypred = clf.predict(xtest)
report = classification_report(ytest, ypred)
print(report)

print(clf.coef_)

# selector = RFE(clf, n_features_to_select=10, step=1)
# selector.fit(xtrain, ytrain)
# print(selector.get_feature_names_out())

# # generate predictions and metrics
# print(gs.best_estimator_)
# ypred_probs = gs.predict_proba(xtest)

# ypred = gs.predict(xtest)

# report = classification_report(ytest, ypred)
# print(report)