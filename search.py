#!/usr/bin/env python3
from pprint import pprint
import os
import argparse
import nltk
import sys

from collections import defaultdict, Counter
from serde import serialize, deserialize

import re
import operator

from math import log10

from functools import reduce

from common import *

stemmer = nltk.stem.PorterStemmer()

def docWeights(term, dictionary, postings):
    if not term in dictionary:
        return []
    count, size, pos = dictionary[term]
    postings.seek(pos)
    docs = deserialize(postings.read(size))
    return docs

def getVec(counts, df, N):
    weights = {}
    for word, count in counts.items():
        if df[word] and count:
            weights[word] = (1 + log10(count)) * log10(N/df[word])
        else:
            weights[word] = 0
    length = vecLength(weights.values())
    norm_weights = {word: weight/length if length else 0 for word, weight in weights.items()}

    return norm_weights

def interactive():
    while True:
        yield input(">>> ")

def queryLines(f):
    for line in f:
        yield line

#================================================
# Main Search Logic
#================================================

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dictionary", required=True)
parser.add_argument("-p", "--postings", required=True)
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-i", "--interactive", action='store_true')

args = parser.parse_args()

postings_file = open(args.postings, 'rb')
dict_file = open(args.dictionary, 'rb')
query_file = open(args.query)
output = open(args.output, 'w')

dictionary = deserialize(dict_file.read())
df = dictionary["DF"]
N = dictionary["N"]

count, size, pos = dictionary["growth"]

postings_file.seek(pos)
growth = deserialize(postings_file.read(size))

if args.interactive:
    query_lines = interactive()
else:
    query_lines = queryLines(query_file)

# Write out the results of each query to the output
for query in query_lines:
    # Heap contains the calculated relevance of the documents for this query
    heap = defaultdict(float)

    # query_vec is a mapping between query terms and their ltc weight
    query_vec = queryVec(query, df, N)

    print(query_vec)
    for word, weight in query_vec.items():
        results = docWeights(word, dictionary, postings_file)
        for doc_id, doc_term_weight in results:
            heap[doc_id] += doc_term_weight * weight

    top10 = sorted(heap.items(), key=lambda x: x[1], reverse=True)[:10]
    out = " ".join(str(i) for i, _ in top10)
    output.write(out + '\n')

    print(out)
