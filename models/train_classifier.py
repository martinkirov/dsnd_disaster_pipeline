import sys
# import libraries
import nltk
nltk.download(['stopwords','punkt', 'wordnet'])
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import pickle 
from sklearn.metrics import f1_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    This function loads the disaster response dataset from the sqlite database.
    IN: database filepath (str)
    OUT: Disaster Response Messages (X), Targets (Y), Category Names (Y.columns)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_table',engine)
    X = df['message']
    Y = df.iloc[:, 3:]
    return X, Y, Y.columns

def tokenize(text):
    """
    tokenize is a custom tokenizer function that:
    - removes punctuation
    - removes english stopwords
    - makes all the text lowercase
    - lemmatizes the words
    IN: string
    OUT: list with stings
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    # all lower case
    processed_text = text.lower()
    # split into words; ignore punctuatuion
    processed_text = tokenizer.tokenize(processed_text)
    # remove stopwords
    processed_text = [w for w in processed_text if w not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    processed_text = [lemmatizer.lemmatize(w, pos='n').strip() for w in processed_text]
    processed_text = [lemmatizer.lemmatize(w, pos='v').strip() for w in processed_text]
    #return
    return processed_text


def build_model():
    """
    Set up a pipeline that:
    - tokenizes the messages
    - TFIDFs them
    - sticks them in a random forrest classifier
    IN: None
    OUT: model object
    """
    count_vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    forest_clf = RandomForestClassifier(n_estimators=10)
    pipeline = Pipeline([
                        ('vectorizer', count_vect),
                        ('tfidf', tfidf),
                        ('clf', MultiOutputClassifier(forest_clf))
                        ])
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """
    Calculate the accuracy, precision, recall and f1 score for each category we are predicting.
    IN: model, X_test (messages), y_test (target categories), category names
    OUT: pandas dataframe with scores (uncomment) OR summary per-metric averages
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.index = X_test.index
    y_pred.columns = category_names
    
    acc_sco = []
    prec_sco = []
    rec_sco = []
    f1_sco = []
    
    for col in category_names:
        acc_sco.append(accuracy_score(y_test[col],y_pred[col]))
        prec_sco.append(precision_score(y_test[col],y_pred[col], average='weighted'))
        rec_sco.append(recall_score(y_test[col],y_pred[col],average='weighted'))
        f1_sco.append(f1_score(y_test[col],y_pred[col], average='weighted'))
        
    output_df = pd.DataFrame()
    output_df['accuracy_score'] = acc_sco
    output_df['precision_score'] = prec_sco
    output_df['recall_score'] = rec_sco
    output_df['f1_score'] = f1_sco
    output_df.index = y_test.columns
    #return output_df
    return output_df.accuracy_score.mean(), output_df.precision_score.mean(), output_df.recall_score.mean(), output_df.f1_score.mean()


def save_model(model, model_filepath):
    """
    Save a pickle instance of the model.
    IN: model object, path for the save
    OUT: None
    """
    file_object = open(model_filepath, 'wb')
    pickle.dump(model, file_object)
    file_object.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        print(evaluate_model(model, X_test, Y_test, category_names))

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