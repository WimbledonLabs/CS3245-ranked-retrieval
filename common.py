import nltk
from math import log10
from collections import Counter

stemmer = nltk.stem.PorterStemmer()

def vecLength(vec):
    squared_length = 0
    for num in vec:
        squared_length += num**2

    return squared_length**0.5

def printDocInfo():
    pass

def cosineNormalize(weights):
    length = vecLength(weights.values())
    if not length:
        return weights

    norm_weights = {word: weight/length for word, weight in weights.items()}
    return norm_weights

def wordStems(sentence):
    for word in nltk.word_tokenize(sentence):
        yield stemmer.stem(word)

def stems(document):
    print(document)
    for sentence in nltk.sent_tokenize(document):
        yield from wordStems(sentence)

def weightsLTC(counts, df, N):
    weights = {}
    for word, count in counts.items():
        if df[word] and count:
            weights[word] = (1 + log10(count)) * log10(N/df[word])
        else:
            weights[word] = 0

    return weights

def weightsLNC(counts):
    pass

def queryVec(query_line, df, N):
    query_terms = [stemmer.stem(word) for word in query_line.strip().lower().split()]

    weights = weightsLTC(Counter(query_terms), df, N)

    return cosineNormalize( weightsLTC(Counter(query_terms), df, N) )
