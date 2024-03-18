# -*- coding: utf-8 -*-
"""
@author: Robin Rai

@file_name: index.py

@description: Builds the Inverted Index Model by processing the raw text of HTML documents
"""
import os
import time
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import regex as re
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer

################################# SETTINGS ###################################

# Change variable "folder_name" to the name of the folder holding the HTMl files
# Alternatively, leave empty ("") to prompt the user for folder name input
folder_name = "videogames"

# Techniques (ON=True/OFF=False):
STOPWORDS = True # Removes top common words such as "the, of, as, a"
LEMMATISATION = True # Reduces different forms of a word to one single form e.g. "building" -> "build"
STEMMING = False # Chops off prefixes/suffixes from words to obtain a common root e.g. "Changing" -> "Chang"
BIGRAMS = True # Groups consecutive words that appear frequently together to capture more contextual information
BIGRAMS_THRESHOLD = 1

# Weighting elements
ELEMENT_WEIGHTING = True # Title, Headers and custom tag elements
CUSTOMTAG = ["gameBioInfo"]

# Weighting adjustments:
TITLEWEIGHT = 3.5
HEADERWEIGHT = 2
CUSTOMWEIGHT = 5

# For evaluation
TIMER = True

##############################################################################

def preprocess(text):
    # regular expression to ignore unwanted characters
    clean_text = re.sub(r'<.+?>|\n', '', text)
    # regular expression to replace hyphens with white space characters
    clean_text = re.sub(r'-', ' ', clean_text)
    # tokenise the text
    tokens = nltk.word_tokenize(clean_text)
    # filter out stopwords (English)
    if STOPWORDS:
        stopwords_english = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords_english]
    else:
        tokens = [word.lower() for word in tokens if word.isalpha()]

    # apply both lemmatisation and stemming
    if LEMMATISATION and STEMMING:
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        tokens = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in tokens]
    # apply just lemmatisation
    elif LEMMATISATION:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    # apply just stemming
    elif STEMMING:
        stemmer = PorterStemmer() # using PorterStemmer
        tokens = [stemmer.stem(word.lower()) for word in tokens]

    token_frequency = Counter(tokens)    

    # detect and add on bigrams to the tokens
    if BIGRAMS:
        detected_bigrams = detectBigrams(tokens)
        token_frequency.update(detected_bigrams)
    
    return token_frequency

def detectBigrams(tokens):
    # Change THRESHOLD value depending on if its indexing or querying
    THRESHOLD = BIGRAMS_THRESHOLD if len(tokens) > 50 else 0
    all_bigrams = nltk.bigrams(tokens)
    bigram_freq = nltk.FreqDist(all_bigrams)

    frequent_bigrams = Counter()
    for bigram, freq in bigram_freq.items():
        if freq > THRESHOLD:
            frequent_bigrams.update([bigram] * freq)

    return frequent_bigrams

def build_inverted_index_model(folder_name):
    documents, docIDs = read_documents(folder_name)

    vocab = {}
    postings = defaultdict(list)
    
    for doc_id, document in enumerate(documents):
        # soupify the document into text
        soup = BeautifulSoup(document, 'html.parser')

        text = soup.get_text()
        
        # call preprocess function
        tokens_freq = preprocess(text)

        # Used for assigning higher weights to these elements
        if ELEMENT_WEIGHTING:
            # Finding title elements 
            title_token = preprocess(soup.title.get_text(separator=' ', strip=True)) if soup.title else ''

            # Finding header elements
            header_texts = [header.get_text(separator=' ', strip=True) for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            header_tokens = Counter()
            for text in header_texts:
                token = preprocess(text)
                header_tokens.update(token)
            
            # Finding custom elements
            custom_tokens = Counter()
            custom_selectors = [f'[class*="{tag}"]' for tag in CUSTOMTAG]
            for selector in custom_selectors:
                custom_text = soup.select(selector)
                
                for text in custom_text:
                    token = preprocess(text.text)
                    custom_tokens.update(token)
            
        # build inverted index model
        for term, freq in tokens_freq.items():
            
            # add to vocab table if new entry
            if term not in vocab:
                vocab[term] = len(vocab)
            term_id = vocab[term]
            # add posting to postings table (including term frequency in all occurences in documents)
            if term_id not in postings:
                postings[term_id] = {}

            weight = 1
            # check if term is a title/header term
            if ELEMENT_WEIGHTING and term in title_token:
                # assign higher weight to title term
                weight += TITLEWEIGHT
            elif ELEMENT_WEIGHTING and term in header_tokens:
                # assign higher weight to header term
                weight += HEADERWEIGHT
            elif ELEMENT_WEIGHTING and term in custom_tokens:
                # assign higher weight to custom term
                weight += CUSTOMWEIGHT
            postings[term_id][doc_id] = (freq, weight)

    return vocab, docIDs, postings

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def read_documents(folder_name):
    script_path = os.path.dirname(os.path.abspath(__file__))
    if not folder_name:
        folder_name = input("Enter folder name to index: ")
    
    folder_path = os.path.join(script_path, folder_name)
    doc_list = [d for d in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, d))]
    pages = []
    doc_ids = {}

    # Initialise counter for doc_id
    doc_id = 0
    counter = 0

    for doc_name in doc_list:
        if doc_name.endswith('.html'):
            counter += 1
            doc_path = os.path.join(folder_path, doc_name)

            # Read HTML file
            with open(doc_path, encoding="utf8") as file:
                html_content = file.read()
                pages.append(html_content)
            
            ## Creating docIDs table:
            # Parse HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Concatenate URL
            url = folder_name + "/" + doc_name

            # Extract contents
            contents_tag = soup.find('meta', {'name': 'description'})
            contents = contents_tag['content'] if contents_tag else "N/A"

            doc_ids[doc_id] = {"url": url, "contents": contents}
            doc_id += 1

    return pages, doc_ids

if __name__ == "__main__":
    print("Indexing...")
    start_time = time.time()
    vocab, docIDs, postings = build_inverted_index_model(folder_name)
    save_to_pickle(vocab, 'vocab.pkl')
    save_to_pickle(docIDs, 'docids.pkl')
    save_to_pickle(postings, 'postings.pkl')
    print("Indexing successful!")

    if TIMER:
        print(f"Time taken: {time.time() - start_time}")

