import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences

#Stop word Removal
stop=set(stopwords.words('english'))
def clean_data(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if not w in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

#Creates our Vector Corpus
def create_vector_corpus(data):
    corpus = []
    for citation in tqdm(data['Citation_Text']):
        words = [word.lower() for word in word_tokenize(citation) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus

#GloVe Embeddings
embedding_dict = {}
with open('glove.6B.100d.txt','r',encoding='utf8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
    f.close()

#Cleaning data columns and replacing the sentiment labels with values
citationDataSet = pd.read_csv("citation_sentiment_corpus.csv",on_bad_lines='skip')
data = citationDataSet.drop(['Source_Paper_ID','Target_Paper_ID'], axis=1)
data['Sentiment'] = data['Sentiment'].replace({'o':0,'p':1,'n':(-1)})
data['Citation_Text'] = data['Citation_Text'].apply(lambda x : clean_data(x))
vectorCorpus = create_vector_corpus(data)

#tokenization of the data
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(vectorCorpus)
sequences = tokenizer_obj.texts_to_sequences(vectorCorpus)

#padding our text input for the model
citation_pad = pad_sequences(sequences,padding='post')

#word index,total number of words, and the construction of our embedding matrix
word_index = tokenizer_obj.word_index
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words,100))
for word, i in tqdm(word_index.items()):
    if i> num_words:
        continue
    emb_vec = embedding_dict.get(word)
    if emb_vec is not None:
        embedding_dict[i]=emb_vec

#Splitting the data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(citation_pad, data.Sentiment, test_size=.3, random_state=1)





