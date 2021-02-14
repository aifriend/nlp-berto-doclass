# df = pd.read_csv("data.csv")
# print(df.head())
# print(df.shape)
# # df.drop_duplicates()
# print(df.shape)
# print(df.columns)
# df.rename(columns={
#         'REPRE': 'Representation'
#     }, inplace=True)
# print(df.columns)
# df.columns = [col.lower() for col in df]
# print(df.columns)
# print(df.isnull().sum())
# df.dropna()
# df.dropna(axis=1)
# repre = df['representation']
# print(repre.head())
# re_boundary_realignment = re.compile('["\\\')\\]}]+?(?:\\s+|(?=--)|$)', re.MULTILINE)
# \-{4,100}\s
# \s\-{4,100}\s
# \s\-\s\-{4,100}
# (\s\-\s\-){4,100}
# \s\-{4,100}
# \-{4,100}\s
# \|
# \([\w\s]*
# --
# - -
# \.\s*?\.
# \[{1,3}\d{1,2}\]{1,3}
# remove email
# remove duplicated white spaces

import nltk
# Import packages and modules
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer  # Create a dataframe

X_train = pd.read_csv("dataset/data_test.csv")
X_train.rename(columns={
    'REPRE': 'speech'
}, inplace=True)


def preprocess_text(text):
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)

    # Lowercase and lemmatise
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]

    # Remove stopwords
    keywords = [lemma for lemma in lemmas if lemma not in stopwords.words('spanish')]
    return keywords


# Create an instance of TfidfVectorizer
vectoriser = TfidfVectorizer(analyzer=preprocess_text)  # Fit to the data and transform to feature matrix
X_train = vectoriser.fit_transform(X_train['speech'])  # Convert sparse matrix to dataframe
X_train = pd.DataFrame.sparse.from_spmatrix(X_train)  # Save mapping on which index refers to which words
col_map = {v: k for k, v in vectoriser.vocabulary_.items()}  # Rename each column using the mapping
for col in X_train.columns:
    print(f"COL: {col}")
    X_train.rename(columns={col: col_map[col]}, inplace=True)
print(X_train)
