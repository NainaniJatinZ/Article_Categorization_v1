import pandas as pd
import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from libretranslatepy import LibreTranslateAPI

st.title("Article Categorization v1")

path_models = "data/"

# SVM
path_svm = path_models + 'best_svc.pickle'
with open(path_svm, 'rb') as data:
    svc_model = pickle.load(data)

path_tfidf = "data/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_text(text):
    
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Content'])
    df.loc[0] = text
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Content_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)    
    lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_5'] = lemmatized_text_list
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    df = df['Content_Parsed_6']
    # df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    df = df.rename('Content_Parsed')
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features

def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category

def predict_from_text(text):
    
    # Predict using the input model
    prediction_svc = svc_model.predict(create_features_from_text(text))[0]
    prediction_svc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_svc = get_category_name(prediction_svc)
    
    print("The predicted category using the SVM model is %s." %(category_svc) )
    print("The conditional probability is: %a" %(prediction_svc_proba.max()*100))
    return category_svc

lt = LibreTranslateAPI("https://translate.argosopentech.com/")


option = st.selectbox('What language is the text in?', ('English', 'Hindi', 'Spanish'))

if option == "Spanish":
    lang = "es"
elif option == "Hindi":
    lang = "hi"
else:
    lang = "en"

text = st.text_area("Article", height=100)
if st.button('Submit'):
    translated = lt.translate(text, lang, "en")
    st.write("Predicted Category: " + predict_from_text(translated))
    st.text_area("Translated to English: ", value=translated)
else:
    st.write('Enter text above')
# st.write("works")


st.title("Details")
st.write("TF-IDF feature extraction (like bag of words but gives more importance to unique words) followed by SVM for categorization")
st.write("Dataset: BBC")
st.write("Present categories: business, entertainment, politics, sport, tech")
st.write("LibreTranslate open source binder for translation")


