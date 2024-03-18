# -*- coding: utf-8 -*-
"""
@author: Robin Rai

@file_name: search.py

@description: Handles the retrieval process of the system using Vector Space Model, 
preprocessing, vectorisation, and ranking
"""
import index
import pickle
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
import math
import numpy as np
import regex as re
from collections import Counter
from fuzzywuzzy import fuzz

################################# SETTINGS ###################################

# Techniques (ON=True/OFF=False):
QUERYEXPANSION = True # Expands inputted query by adding similar terms to minimise query-document mismatch
SPELLINGCORRECTION = True # Corrects spelling errors to closest recognisable term in the collection of documents (Automatic Thesaurus Generation)

# Weighting adjustments:
QUERYEXPANDEDTERMS = 0.4 # Weighting of terms occurring from query expansion compared to the original search terms

# Other:
PRECISION = 10 # (precision@10 DEFAULT) how many results are retrieved and shown 

# For evaluation
SHOWSCORE = False # Display weighted TF-IDF score value of each document retrieved

##############################################################################

def load_from_pickle():
    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open("docids.pkl", 'rb') as f:
        docIDs = pickle.load(f)
    with open("postings.pkl", 'rb') as f:
        postings = pickle.load(f)
    
    return vocab, docIDs, postings

# Function to calculate TF-IDF (weighted)
def calculate_tf_idf(tf, df, N, term_weight=1.0):
    # Calculate TF-IDF using the formula: (1 + log(tf) * log(N/df))
    tfidf = ((term_weight * (1 + math.log(tf))) * math.log(N/df)) if tf > 0 and df > 0 else 0
    return tfidf

# Function process for calculating TF-IDF for query 
def calculate_query_tfidf(tokens, term_id, term, postings, N):
    tf = tokens[term]
    df = len(postings.get(term_id, {}))  # Number of documents containing the term
    return calculate_tf_idf(tf, df, N)

# Function process for calculating TF-IDF for documents
def calculate_document_tfidf(doc_id, term_id, postings, N, term_weight=1.0):
    # tfTuple is a tuple containing term frequency [0] and term weight [1]
    tfTuple = postings[term_id].get(doc_id, 0)
    
    if tfTuple != 0:
        tf = tfTuple[0]
        term_weight = tfTuple[1]
    else:
        tf = 0

    df = len(postings[term_id])
    

    return calculate_tf_idf(tf, df, N, term_weight)

# Euclidean distance calculator
# (Squares all the tf, summing them, and then square rooting the sum)
def eucLength(matrix):
    return math.sqrt(sum([x**2 for x in matrix]))

# Function to calculate cosine similarity between two TF-IDF vectors
def cosine_similarity(query_tfidf, document_tfidf):
    # calculate dot product between the vectors.
    num = np.dot(query_tfidf, document_tfidf) 

    # Magnitude of the TF-IDF vectors
    query_norm = eucLength(query_tfidf)
    document_norm = eucLength(document_tfidf)

    denom = query_norm * document_norm

    return num / denom if query_norm != 0 and document_norm != 0 else 0 # return similarity

# Expand query with similar terms for query terms that are 
# part-of-speech (pos) tagged as nouns
def expand_query(query, posTags):
    # set stop words
    stops = set(stopwords.words('english'))
    # filter out stop words
    cleanedTags = [t for t in posTags if t[0] not in stops]
    # keep only noun tags
    nounTags = [t[0] for t in cleanedTags if t[1].startswith('NN')]
    synonyms = []
    for noun in nounTags:
        synsets = wordnet.synsets(noun)
        if synsets:
            lemma_names = []
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name() not in nounTags and lemma.name() not in lemma_names:
                        lemma_names.append(lemma.name())
            synonyms.extend(lemma_names)
    
    modified_synonyms = ' '.join(synonyms)

    # return list of expanded terms
    return modified_synonyms

# part-of-speech tag query terms
def pos_tag_query(query):
    query_tokens = word_tokenize(query)
    posTags = pos_tag(query_tokens)
    return posTags

# calculates Levenshtein's distance and accounts for size
# differences between misspelt term and terms in the 'vocab' dictionary
def calculate_token_similarity(query_token, vocab):
    highest_val = 0
    highest_word = None
    for word in vocab:
        ratio = fuzz.ratio(query_token, word)
        size_difference_penalty = abs(len(query_token) - len(word))
        score = ratio - size_difference_penalty

        if score > highest_val:
            highest_val = score
            highest_word = word
        
    return highest_word

# begin process for spelling correction on given query terms
def correct_spelling(processed_query_tokens):
    corrected_tokens = Counter()

    for token in processed_query_tokens:
        # if mispelt or unknown term then correct spelling
        if not token in vocab.keys():
           correct_word = calculate_token_similarity(token, vocab)
           corrected_tokens[correct_word] = processed_query_tokens[token] 
        else:
            corrected_tokens[token] = processed_query_tokens[token]
            
    return corrected_tokens

def retrieve_top_k(query, k=10):
    # run preprocessing on query 
    # ->(tokenisation/stemming/lemmitisation/stopwords/bigrams/etc.)
    query_tokens = index.preprocess(query)

    # perform spelling correction
    if SPELLINGCORRECTION:
        query_tokens = correct_spelling(query_tokens)

    # perform query expansion
    if QUERYEXPANSION:
        posTags = pos_tag_query(query)
        synonyms = expand_query(query, posTags)
        expanded_terms = index.preprocess(synonyms)
        # lower weight of terms occuring from expansion
        for term in expanded_terms:
            expanded_terms[term] = 1 * QUERYEXPANDEDTERMS
        query_tokens += expanded_terms 
    
    query_tfidf = [
        calculate_query_tfidf(query_tokens, term_id, term, postings, len(docIDs))
        for term, term_id in vocab.items()
    ]

    document_tfidf = [
        [
            calculate_document_tfidf(doc_id, term_id, postings, N)
            for term_id in postings.keys()
        ]
        for doc_id in range(N)
    ]

    similarity_scores = {
        doc_id: cosine_similarity(query_tfidf, document_tfidf[doc_id])
        for doc_id in range(N)
    }

    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)    
    return ranked_documents[:k]

# Displays list of tuples of ranked documents with their weighting score as: Rank number, URL, Content
def show_top_k(ranked_docs):
    print(f"\nTop {len(ranked_docs)} Documents:")
    for index, doc in enumerate(ranked_docs):
        doc_id = doc[0]
        url = docIDs[doc_id]["url"]
        contents = docIDs[doc_id]["contents"]
        if SHOWSCORE:
            print(f"{index+1}. URL:{url} score: {doc[1]}\n\nDescription:\n{contents}\n\n")
        else:
            print(f"{index+1}. URL:{url}\n\nDescription:\n{contents}\n\n")


vocab, docIDs, postings = load_from_pickle()
N = len(docIDs) # Total number of documents

if __name__ == "__main__":
    while True:
        query = input("Enter your query (type 'quit' to exit):\n")
        if query == "quit":
            break
        else:
            # regular expression to replace hyphens with white space characters
            query = re.sub(r'-', ' ', query)
            ranked_docs = retrieve_top_k(query, PRECISION)
            show_top_k(ranked_docs)

            if SHOWSCORE: # for evaluation
                for doc in ranked_docs:
                    print(doc)
            



