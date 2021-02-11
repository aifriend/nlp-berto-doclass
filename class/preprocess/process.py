from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from lingualytics.preprocessing import remove_lessthan, remove_punctuation, remove_stopwords
from lingualytics.stopwords import hi_stopwords, en_stopwords
from texthero.preprocessing import remove_digits

df = pd.read_csv(
    "dataset/data_clean.csv",
    header=None, sep=',', names=['ID', 'LENTGH', 'CONTENT']
)
pd.set_option('display.max_colwidth', None)
df['clean_text'] = df['CONTENT'].pipe(remove_digits) \
    .pipe(remove_punctuation) \
    .pipe(remove_lessthan, length=3) \
    .pipe(remove_stopwords, stopwords=en_stopwords.union(hi_stopwords))

for st in df['CONTENT']:
    print(st)

from lingualytics.learner import Learner

learner = Learner(model_type='bert',
                  model_name='bert-base-multilingual-cased',
                  data_dir='dataset')
learner.fit()
