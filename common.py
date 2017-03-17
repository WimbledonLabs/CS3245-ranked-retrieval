import nltk
from math import log10
from collections import Counter

stemmer = nltk.stem.PorterStemmer()

def vecLength(vec):
    """ Compute sqrt(v[0]^2 + v[1]^2 + ...) """
    squared_length = 0
    for num in vec:
        squared_length += num**2

    return squared_length**0.5

def cosineNormalize(weights):
    """ Normalize the given map of stems to numbers so that the length is 1 """
    length = vecLength(weights.values())
    if not length:
        return weights

    norm_weights = {word: weight/length for word, weight in weights.items()}
    return norm_weights

def wordStems(sentence):
    """ Yield stemed words for the provided sentence """
    for word in nltk.word_tokenize(sentence):
        yield stemmer.stem(word)

def stems(document):
    """ Yield stemed words for the provided document """
    for sentence in nltk.sent_tokenize(document):
        yield from wordStems(sentence)

def weightsLTC(counts, df, N):
    """ Calculate the weight of words counts using tf-idf weighting and
    consine normalization """
    weights = {}
    for word, count in counts.items():
        if df[word] and count:
            weights[word] = (1 + log10(count)) * log10(N/df[word])
        else:
            weights[word] = 0

    return cosineNormalize(weights)

def weightsLNC(counts):
    """ Calculate the weight of words counts using tf weighting and cosine
    normalization """
    weights = {}
    for word, count in counts.items():
        weights[word] = (1 + log10(count))

    return cosineNormalize(weights)

def queryVec(query_line, df, N):
    """ Return a vector representing the given query in the vector space
    model"""
    query_terms = list(stems(query_line.strip().lower()))
    return weightsLTC(Counter(query_terms), df, N)
