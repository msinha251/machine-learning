import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from ml.data import basic_preprocess


def train_model(df, target):
    '''
    Train model and return it
    '''
    # Clean data
    df, cat_cols, num_cols, target = basic_preprocess(df, train=True, target=target)

    # Final features
    cat_cols_to_use = ['homeplanet', 'cabin', 'destination']
    final_cols = cat_cols_to_use + list(num_cols)

    # Split data into train and validation
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)

    # Split data into X and y
    X_train_df = df_train.drop(target, axis=1)
    y_train = df_train[target]

    X_val_df = df_val.drop(target, axis=1)
    y_val = df_val[target]

    # convert to dicts
    train_dicts = X_train_df[final_cols].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)

    # transform dicts
    X_train = dv.transform(train_dicts)
    val_dicts = X_val_df[final_cols].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1, oob_score=True)
    model.fit(X_train, y_train)

    # Predict on validation data
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = y_pred_prob > 0.5

    # Calculate metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)

    # Print metrics
    print('***** Validation metrics *****')
    print(f"AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print('******************************')

    return model, dv

def predict_batch(df, model, dv, target):
    '''
    Predict on df using model and dv
    '''
    # Clean data
    df, cat_cols, num_cols, target = basic_preprocess(df, train=False, target=target)

    # Final features
    cat_cols_to_use = ['homeplanet', 'cabin', 'destination']
    final_cols = cat_cols_to_use + list(num_cols)

    # convert to dicts
    dicts = df[final_cols].to_dict(orient='records')

    # transform dicts
    X = dv.transform(dicts)

    # Predict
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = y_pred_prob > 0.5

    # Add predictions to df
    df['prediction'] = y_pred
    df['prediction_probability'] = y_pred_prob

    return df


def save_model(model, dv, path):
    '''
    Save model and dv to path
    '''
    logging.info(f'Saving model to {path}')
    with open(path, 'wb') as f:
        pickle.dump((model, dv), f)
        
def load_model(path):
    '''
    Load model and dv from path
    '''
    logging.info(f'Loading model from {path}')
    with open(path, 'rb') as f:
        model, dv = pickle.load(f)
    return model, dv





