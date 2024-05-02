import pickle
import os
import pandas as pd
messages = pd.read_csv('train_spam.csv')
import seaborn as sns
import matplotlib.pyplot as plt
# sns.countplot(x='text_type', data=messages)
# plt.show()

import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def data_preproc():
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    if os.path.exists('processed_corpus.pickle'):
        with open('processed_corpus.pickle', 'rb') as file:
            corpus = pickle.load(file)
            return corpus
    else:
        corpus = []
        for i in range(0, len(messages)):
            review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
            review = review.lower()
            review = review.split()
            review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        with open('processed_corpus.pickle', 'wb') as handle:
            pickle.dump(corpus, handle)
            return corpus
def data_prepare(corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 3500)
    X = cv.fit_transform(corpus).toarray()

    y = pd.get_dummies(messages['text_type'])
    y = y.iloc[:, 1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    return X_train, X_test, y_train, y_test
def fit_model(X_train,y_train):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB(alpha=0.8)
    mnb.fit(X_train,y_train)
    # print(mnb.predict(X_test))

def get_embeddings():
    import torch.nn.functional as F

    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel

    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Each input text should start with "query: " or "passage: ".
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    input_texts = data_preproc()[:10]

    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2')

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

if __name__ == "__main__":
    embeddings = get_embeddings()
    print(embeddings)
