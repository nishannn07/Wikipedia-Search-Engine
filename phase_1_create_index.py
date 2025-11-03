#!/usr/bin/env python
# coding: utf-8

import re
import os
import sys
import nltk
import xml.etree.cElementTree as et
import pickle
import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# --- Setup (from both scripts) ---

# 1. Download and set up stop words (from your friend's script)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 2. Set up stemmer (from your original script)
stemmer = SnowballStemmer('english')

# 3. Set up regex patterns (from your original script)
pattern = re.compile("[^a-zA-Z0-9]")
cssExp = re.compile(r'{\|(.*?)\|}', re.DOTALL)
linkExp = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.DOTALL)
categoryExp = re.compile(r"\[\[Category:(.*?)\]\]", re.DOTALL)
infoboxExp = re.compile(r"{{Infobox((.|\n)*?)}}", re.DOTALL)


# --- Helper Functions ---

def clean_and_tokenize(text):
    """
    Cleans, tokenizes, stems, and removes stop words from a string.
    """
    final_tokens = []
    text = text.lower()
    words = re.split(pattern, text)
    for word in words:
        word = word.strip()
        if len(word) <= 2:  # Remove short tokens
            continue
        if word and word not in stop_words:
            stemmed_word = stemmer.stem(word)
            final_tokens.append(stemmed_word)
    return final_tokens

def write_into_file(filename, inverted_object, flag, document_word_dict):
    """
    Writes the in-memory inverted index to a file and records
    the file pointer (byte offset) for each word.
    (This is from your original script)
    """
    fileptr = open(filename, "w+")
    pointer = 0
    for word in sorted(inverted_object.keys()): # Sorting is good for merging later
        posting_list = ",".join(inverted_object[word])
        posting_list = posting_list + "\n"
        
        if word not in document_word_dict:
            document_word_dict[word] = {}
        document_word_dict[word][flag] = pointer
        
        fileptr.write(posting_list)
        pointer += len(posting_list.encode('utf-8')) # Use bytes for accurate offset
    fileptr.close()
    return document_word_dict

def write_pickle_file(filename, pickleobj):
    """Writes an object to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(pickleobj, file)

def count_frequencies(token_list):
    """Helper to count token frequencies in a list."""
    freq_dict = {}
    for word in token_list:
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1
    return freq_dict


# --- Main Indexing Function ---

def create_index(wikipedia_dump, index_path):
    """
    Parses the Wikipedia XML and builds the inverted index files.
    """
    
    # In-memory inverted indexes
    title_inverted_index = {}
    body_inverted_index = {}
    category_inverted_index = {}
    infobox_inverted_index = {}

    # Dictionaries for document IDs and word file pointers
    document_title = {}
    document_word = {}
    document_no = 0
    
    start = time.time()
    
    # Use iterparse for memory-efficient XML parsing
    content = et.iterparse(wikipedia_dump, events=("start", "end"))
    content = iter(content)

    print("Starting XML parse and indexing...")

    for event, context in content:
        tag = re.sub(r"{.*}", "", context.tag)
        
        if event == "end":
            if tag == "title":
                title_text = context.text
            
            elif tag == "text":
                body_text = context.text
            
            elif tag == "page":
                d_no = str(document_no)
                document_title[document_no] = title_text or "" # Store original title
                
                # 1. Process Title
                if title_text:
                    title_tokens = clean_and_tokenize(title_text)
                    title_freq = count_frequencies(title_tokens)
                    for word, freq in title_freq.items():
                        if word not in title_inverted_index:
                            title_inverted_index[word] = []
                        title_inverted_index[word].append(f"{d_no}:{freq}")
                
                if body_text:
                    # 2. Process Category (using friend's regex)
                    categories = categoryExp.findall(body_text)
                    category_text = ' '.join(categories)
                    category_tokens = clean_and_tokenize(category_text)
                    category_freq = count_frequencies(category_tokens)
                    for word, freq in category_freq.items():
                        if word not in category_inverted_index:
                            category_inverted_index[word] = []
                        category_inverted_index[word].append(f"{d_no}:{freq}")

                    # 3. Process Infobox (using your regex)
                    info_words = infoboxExp.findall(body_text)
                    infobox_text = ' '.join([i[0] for i in info_words])
                    infobox_tokens = clean_and_tokenize(infobox_text)
                    infobox_freq = count_frequencies(infobox_tokens)
                    for word, freq in infobox_freq.items():
                        if word not in infobox_inverted_index:
                            infobox_inverted_index[word] = []
                        infobox_inverted_index[word].append(f"{d_no}:{freq}")
                                        
                    # 4. Process Body Text (cleaning first)
                    body_text = linkExp.sub('', body_text)
                    body_text = cssExp.sub('', body_text)
                    body_tokens = clean_and_tokenize(body_text)
                    body_freq = count_frequencies(body_tokens)
                    for word, freq in body_freq.items():
                        if word not in body_inverted_index:
                            body_inverted_index[word] = []
                        body_inverted_index[word].append(f"{d_no}:{freq}")
                
                # Progress update
                if document_no % 10000 == 0:
                    print(f"  ...processed {document_no} documents.")
                
                document_no += 1
                
            # Clear context to save memory
            context.clear()

    end_parse = time.time()
    print(f"\nParsing complete. Total documents: {document_no}")
    print(f"Time taken for parsing: {end_parse - start:.2f} seconds")

    # --- Write index files ---
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    
    print("Writing index files...")
    
    filename = os.path.join(index_path, "title.txt")
    document_word = write_into_file(filename, title_inverted_index, 't', document_word)
    
    filename = os.path.join(index_path, "category.txt")
    document_word = write_into_file(filename, category_inverted_index, 'c', document_word)
    
    filename = os.path.join(index_path, "infobox.txt")
    document_word = write_into_file(filename, infobox_inverted_index, 'i', document_word)
    
    filename = os.path.join(index_path, "body_text.txt")
    document_word = write_into_file(filename, body_inverted_index, 'b', document_word)

    # --- Write pickle files ---
    print("Writing pickle files...")
    
    filename = os.path.join(index_path, "word_position.pickle")
    write_pickle_file(filename, document_word)
    
    filename = os.path.join(index_path, "title_doc_no.pickle")
    write_pickle_file(filename, document_title)

    end_total = time.time()
    print(f"\nAll done. Total time: {end_total - start:.2f} seconds")

# --- Main execution ---
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python phase_1_create_index.py <path_to_wiki.xml> <path_to_index_dir>")
        sys.exit(1)
        
    wikipedia_dump = sys.argv[1]
    index_path = sys.argv[2]
    
    create_index(wikipedia_dump, index_path)
