import os
import json
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA")
TEST_DATA_DIR = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

with open(os.environ.get("PARAMS")) as json_params:
    PARAMS = json.load(json_params)

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

def main():
    ## Read in data, perform basic cleaning
    test_df = pd.read_csv(TEST_DATA_DIR)
    df = pd.read_csv(TRAINING_DATA_DIR)
    df = drop_columns(df)
    convert_categoricals_to_num(df)

    ## Split data
    train_df, ytrain, valid_df, yvalid = split_data(df)

    ## Clean data, train model
    clf = build_pipeline(train_df)
    clf.fit(train_df, ytrain)

    ## Test model on validation set
    # preds = clf.predict_proba(valid_df)[:, 1]
    preds = clf.predict(valid_df)
    print(metrics.roc_auc_score(yvalid, preds))

    # joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    # joblib.dump(imputer, f"models/{MODEL}_{FOLD}_imputer.pkl")
    # joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    # joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

def convert_categoricals_to_num(df):
    """Converts all categorical object types to numerical representation"""
    for c in PARAMS['categoricals']:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes.astype('category')

def split_data(df):
    try:
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold==FOLD].reset_index(drop=True)
    except AttributeError:
        train_df, valid_df = train_test_split(df, test_size=0.2)

    # ytrain = train_df.target.values
    # yvalid = valid_df.target.values
    ytrain = train_df[PARAMS['target']]
    yvalid = valid_df[PARAMS['target']]
    train_df = train_df.drop([PARAMS['target']], axis=1)
    valid_df = valid_df.drop([PARAMS['target']], axis=1)

    return (train_df, ytrain, valid_df, yvalid)

def drop_columns(df):
    df = df.drop(PARAMS['dropFeatures'], axis=1)
    return df

def build_pipeline(train_df):
    imputer = build_imputer(train_df)
    encoder = build_encoder(train_df)

    clf = Pipeline(steps = [('imputer', imputer), 
                            ('encoder', encoder),
                            ('classifier', dispatcher.MODELS[MODEL])])
    return clf

def build_imputer(train_df):
    cat_vars = PARAMS['categoricals']
    num_vars = [c for c in train_df.columns if c not in cat_vars] 
    cat_imp = ('cat_impute', SimpleImputer(strategy='most_frequent'), cat_vars)
    num_imp = ('num_impute', SimpleImputer(strategy='mean'), num_vars)
    return ColumnTransformer(transformers = [cat_imp, num_imp], remainder='passthrough')

def build_encoder(train_df):
    cat_var_idxs = [list(train_df.columns).index(c) for c in PARAMS['categoricals']]
    enc = ('encode', OneHotEncoder(handle_unknown='ignore'), cat_var_idxs)
    return ColumnTransformer(transformers = [enc], remainder='passthrough')


if __name__ == "__main__":
    main()