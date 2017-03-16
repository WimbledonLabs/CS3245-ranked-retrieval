#!/usr/bin/env python3
from pprint import pprint
import os
import argparse
import nltk
import operator

from collections import defaultdict, Counter
from functools import reduce

from math import log10


from serde import serialize, deserialize

stemmer = nltk.stem.PorterStemmer()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--source", required=True)
parser.add_argument("-d", "--dictionary", required=True)
parser.add_argument("-p", "--postings", required=True)

args = parser.parse_args()
document_dir = args.source

# Index is set a a defaultdict(set) so we never have to explicitly check if
# a key in index exists before adding a document to it
index = defaultdict(set)

def vecLength(vec):
    squared_length = 0
    for num in vec:
        squared_length += num**2

    return squared_length**0.5

def stems(document):
    for sentence in nltk.sent_tokenize(doc):
        for word in nltk.word_tokenize(sentence):
            yield stemmer.stem(word)

#================================================
# Main Indexing Code
#================================================

doc_list = os.listdir(document_dir)
N = len(doc_list)

document_counts = {}
df = defaultdict(int)

# Iterate through the files in the provided directory, and create a map between
# terms and the documents they appear in
for document_name in doc_list:
    with open(os.path.join(document_dir, document_name)) as document:
        doc = document.read().lower()
        doc_id = int(document_name)

        word_count = Counter(stems(document))
        document_counts[doc_id] = word_count

        for word in word_count:
            df[word] += 1

document_weights = defaultdict(list)
index = defaultdict(set)

for doc_id, counts in document_counts.items():
    weights = {}
    for word, count in counts.items():
        weights[word] = (1 + log10(count)) * log10(N/df[word])
    length = vecLength(weights.values())
    norm_weights = {word: weight/length for word, weight in weights.items()}

    for word, norm_weight in norm_weights.items():
        index[word].add( (doc_id, norm_weight) )

# Create a dictionary which stores the data necessary to retrieve
# the posting for the term
dictionary = {}
dictionary["DF"] = df
dictionary["N"] = N

with open(args.postings, 'wb') as postings_file:
    for word, docs in index.items():
        s = serialize(docs)
        dictionary[word] = (len(docs), len(s), postings_file.tell())
        postings_file.write(s)

# Write the dictionary to the specified dictionary file
with open(args.dictionary, 'wb') as dict_file:
    dict_file.write(serialize(dictionary))

print("GOGOGO!")
