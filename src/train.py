import os
import json
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA")
TEST_DATA_DIR = os.environ.get("TEST_DATA")
PARAMS_DIR = os.environ.get("PARAMS")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    with open(PARAMS_DIR) as json_params:
        params = json.load(json_params)
    df = pd.read_csv(TRAINING_DATA_DIR)
    test_df = pd.read_csv(TEST_DATA_DIR)

    try:
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold==FOLD].reset_index(drop=True)
    except AttributeError:
        train_df, valid_df = train_test_split(df, test_size=0.2)

    # ytrain = train_df.target.values
    # yvalid = valid_df.target.values
    target = params['target']
    ytrain = train_df[target]
    yvalid = valid_df[target]

    # train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    # valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)
    train_df = train_df.drop([params['id'], params['target']], axis=1)
    valid_df = valid_df.drop([params['id'], params['target']], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    print(train_df.dtypes)
    for c in train_df.columns:
        print(c)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + test_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
