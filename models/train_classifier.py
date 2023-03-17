import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

import pickle

def load_data(database_filepath, table_name):
    '''
    :param database_filepath: filepath to SQL db
    :param table_name: table name to load as dataframe
    :return: X independent variable "X" (inputs) for ML
             y dependent “y” variable (output) for ML
             category_cols categories for ML
    '''
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    column_names = list(df.columns)
    category_cols = column_names[4:]
    # define features and set independent variable "X" (inputs) and dependent “y” variable (output)
    X = df.message
    y = df[category_cols]
    return (X, y, category_cols)

def tokenize(text):
    '''
    This function is adopted from Udacityy courcework "Data Scientist""
    :param text: text to tokenize
    :return: tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return (clean_tokens)

def build_model():
    '''
    This model is adopted from Udacityy courcework "Data Scientist"". Ckassifier is parametrized based on the model tuning outcome.  
    :param: None
    :return: the pipeline for bulbing th emodel
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, max_depth=None)))
    ])
    return (pipeline)


def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function prints the classification report for the given model andsaves it in a .csv file
    :param model: model to evaluate
    :param X_test: test input data for ML
    :param y_test: test dependent data for ML
    :param category_names: data columns for the ML feature(s)
    :return: None
    '''
    y_pred = model.predict(X_test)
    clas_rep = classification_report(y_test, y_pred, target_names=category_names)
    print(clas_rep)
    save_clas_rep(clas_rep)
    

def save_clas_rep(clas_rep):
    '''
    This function saves classification report for visualization in a .csv file 
    # Save classification report for vizualization
    
    Adopted from source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
    Alternative: use classification_report(y_test, y_pred, output_dict=True) - available in scikit-learn 0.20.0
    
    :param clas_rep: classification report
    :return: None
    '''
    report_data = []
    lines = clas_rep.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df_clas_rep = pd.DataFrame.from_dict(report_data)
    df_clas_rep.to_csv('models/clas_rep.csv', index = False)



def save_model(model, model_filepath):
    '''
    :param model: ML model to store a .pkl file
    :param model_filepath: filepath to store ML model
    :return: None
    '''
    pickle.dump(model, open(
        model_filepath, 'wb'))


def main():
    if len(sys.argv) == 4:
        database_filepath, table_name, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {} TABLE: {}'.format(database_filepath, table_name))
        X, y, category_names = load_data(database_filepath, table_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()