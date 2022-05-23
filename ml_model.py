import re, string, nltk
from nltk.corpus import stopwords
import numpy as np
import pickle

with open('bin/freqs.chat', 'rb') as handle:
    freqs = pickle.load(handle)

with open('bin/theta.chat', 'rb') as handle:
    theta = pickle.load(handle)

def process_string(phrase):
    
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # Clean data
    
    phrase = re.sub(r'\$\w*', '', phrase)
    phrase = re.sub(r'^RT[\s]+', '', phrase)
    phrase = re.sub(r'https?:\/\/.*[\r\b]*', '', phrase)
    phrase = re.sub(r'#', '', phrase)
    phrase = re.sub('(.)\\1{2,}','\\1', phrase)
    
    # Tokenize data
    
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    phrase_tokens = tokenizer.tokenize(phrase)
    
    # Stemmer
    
    phrase_clean = []
    
    for word in phrase_tokens:
        
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            phrase_clean.append(stem_word)
            
    return phrase_clean

# Extract features

def get_features(phrase, freqs):
    
    word_one = process_string(phrase)
    x = np.zeros((1, 3))
    
    # Bias term is set to 1
    x[0, 0] = 1
    
    for word in word_one:
        # Increment the word count for thepositive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)
        # Incement the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)
    
    assert(x.shape == (1, 3))
    return x

# Sigmoid function

def sigmoid(z):
    
    zz = np.negative(z)
    
    h = 1 / (1 + np.exp(zz))
    
    return h



def predict_phrase(phrase, freqs, theta):
    
    # get the features
    
    x = get_features(phrase, freqs)
    ypred = sigmoid(np.dot(x, theta))
    
    return ypred


# Predict with phrase

def pre(sentence):
    yhat = predict_phrase(sentence, freqs, theta)
    if yhat > 0.5:
        return 'positive'
    elif yhat == 0:
        return 'neutral'
    else:
        return 'negative'

