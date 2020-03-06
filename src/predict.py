import os
import numpy as np
import pandas as pd
import json
import joblib
import csv

# from . import dispatcher

TEST_DATA_DIR = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
with open(os.environ.get("PARAMS")) as json_params:
    PARAMS = json.load(json_params)

def main():
    submission = predict()
    if not os.path.exists('models'):
        os.mkdir('models')
    submission.to_csv(f"models/{MODEL}.csv", index=False)

def predict():
    """
    input
    output
    logic
    TODO: Add CV functionality
    """
    FOLD = 0
    test_df = pd.read_csv(TEST_DATA_DIR)
    idxs = test_df[PARAMS['id']]

    convert_categoricals_to_num(test_df)

    train_cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
    test_df = test_df[train_cols]

    clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_pipe.pkl"))
    preds = clf.predict(test_df)

    sub = pd.DataFrame(np.column_stack((idxs, preds)), columns=["id", "target"])
    return sub

    # for FOLD in range(5):
    #     print(FOLD)
    #     df = pd.read_csv(TEST_DATA_DIR)
    #     encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
    #     cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
    #     for c in encoders:
    #         print(c)
    #         lbl = encoders[c]
    #         df.loc[:, c] = lbl.transform(df[c].values.tolist())

    #     # data is ready to train
    #     clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))

    #     df = df[cols]
    #     preds = clf.predict_proba(df)[:, 1]

    #     if FOLD == 0:
    #         predictions = preds
    #     else:
    #         predictions += preds

    # predictions /= 5

    # sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    # return sub

def convert_categoricals_to_num(df):
    """Converts all categorical object types to numerical representation"""
    for c in PARAMS['categoricals']:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes.astype('category')

if __name__ == "__main__":
    main()
