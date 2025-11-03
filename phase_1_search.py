import sys
import re
import os
import nltk
import pickle
import copy
import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# --- Setup (MUST MATCH create_index.py) ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
pattern = re.compile("[^a-zA-Z0-9]")
# --- End of Setup ---


def clean_and_tokenize(text):
    """
    Cleans, tokenizes, stems, and removes stop words from a query.
    This function MUST be identical to the one in the indexer.
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


def read_file(testfile):
    with open(testfile, 'r') as file:
        queries = file.readlines()
    return queries


def write_file(outputs, path_to_output):
    '''outputs should be a list of lists.
        len(outputs) = number of queries
        Each element in outputs should be a list of titles corresponding to a particular query.'''
    with open(path_to_output, 'w') as file:
        for output in outputs:
            for line in output:
                file.write(line.strip() + '\n')
            file.write('\n')


def search(path_to_index, queries):
    
    # --- Load Index Files ---
    try:
        fields = {}
        fields["t"] = open(os.path.join(path_to_index, "title.txt"), "r")
        fields["c"] = open(os.path.join(path_to_index, "category.txt"), "r")
        fields["i"] = open(os.path.join(path_to_index, "infobox.txt"), "r")
        fields["b"] = open(os.path.join(path_to_index, "body_text.txt"), "r")

        with open(os.path.join(path_to_index, "word_position.pickle"), "rb") as f:
            words_position = pickle.load(f)
        
        with open(os.path.join(path_to_index, "title_doc_no.pickle"), "rb") as f:
            title_position = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error: Could not load index files from '{path_to_index}'.")
        print(f"Missing file: {e.filename}")
        print("Please ensure you have run the indexer first.")
        return []
        
    print("Index files loaded successfully.")
    final_query_result = []

    for query in queries:
        query_result_docs = []
        if (query.strip() == "quit"):
            break
            
        start = time.time()
        query = query.lower().strip()
        
        # This will hold the doc IDs for this query
        doc_id_set = set()
        
        # Check if it's a fielded query
        if ":" in query:
            query_parts = query.split()
            field_queries = {} # To store tokens for each field
            
            for part in query_parts:
                if ":" in part:
                    try:
                        field, term = part.split(":", 1)
                        if field == "ref" or field == "ext" or field == "body":
                            field = "b"
                        elif field == "title":
                            field = "t"
                        elif field == "category":
                            field = "c"
                        elif field == "infobox":
                            field = "i"
                        else:
                            # If field is invalid, treat as normal word
                            term = part 
                            field = None
                            
                    except ValueError:
                        term = part # Not a valid field query
                        field = None
                else:
                    term = part
                    field = None # No field specified
                
                # Tokenize the term
                tokens = clean_and_tokenize(term)
                
                # If no field, add to all fields
                if field is None:
                    for f in fields.keys():
                        if f not in field_queries:
                            field_queries[f] = []
                        field_queries[f].extend(tokens)
                # If field is specified, add only to that field
                elif field in fields.keys():
                    if field not in field_queries:
                        field_queries[field] = []
                    field_queries[field].extend(tokens)

            # --- *** FIXED Fielded Search Logic *** ---
            temp_results = []
            first_field_processed = False

            for field, tokens in field_queries.items():
                if not tokens:
                    continue
                
                # Get results for all tokens *within this field*
                docs_for_this_field = set()
                first_token = True
                
                for word in tokens:
                    word_docs = set()
                    if word in words_position and field in words_position[word]:
                        pointer = words_position[word][field]
                        fields[field].seek(pointer)
                        posting_list = fields[field].readline().strip()
                        
                        for doc in posting_list.split(","):
                            doc_id = doc.split(":")[0]
                            if doc_id:
                                word_docs.add(doc_id)
                    
                    # Intersect results for tokens *within* the same field
                    if first_token:
                        docs_for_this_field = word_docs
                        first_token = False
                    else:
                        docs_for_this_field.intersection_update(word_docs)
                
                # Intersect results *across* different fields
                if not first_field_processed:
                    temp_results = docs_for_this_field
                    first_field_processed = True
                else:
                    temp_results.intersection_update(docs_for_this_field)

            doc_id_set = temp_results

        else:
            # --- *** FIXED Non-Fielded Search Logic *** ---
            query_tokens = clean_and_tokenize(query)
            
            temp_results = []
            first_token_processed = False
            
            for word in query_tokens:
                docs_for_this_word = set()
                if word in words_position:
                    # Search in all fields
                    for field in words_position[word].keys():
                        pointer = words_position[word][field]
                        fields[field].seek(pointer)
                        posting_list = fields[field].readline().strip()
                        
                        for doc in posting_list.split(","):
                            doc_id = doc.split(":")[0]
                            if doc_id:
                                docs_for_this_word.add(doc_id)
                
                # Intersect results (AND logic)
                if not first_token_processed:
                    temp_results = docs_for_this_word
                    first_token_processed = True
                else:
                    temp_results.intersection_update(docs_for_this_word)
            
            doc_id_set = temp_results

        # --- Convert Doc IDs to Titles ---
        final_titles = []
        for doc_id in doc_id_set:
            try:
                # doc_id is a string, title_position keys are integers
                title = title_position[int(doc_id)]
                final_titles.append(title)
            except (KeyError, ValueError):
                # This can happen if doc_id is empty or invalid
                pass
        
        end = time.time()
        print(f"Query: '{query.strip()}' -> Found {len(final_titles)} results in {end - start:.4f} seconds")

        if len(final_titles) > 10:
            final_titles = final_titles[0:10]
            
        final_query_result.append(final_titles)

    # Close all field files
    for f in fields.values():
        f.close()

    return final_query_result



def main():
    if len(sys.argv) != 4:
        print("Usage: python phase_1_search.py <path_to_index> <testfile> <path_to_output>")
        sys.exit(1)
        
    path_to_index = sys.argv[1]
    testfile = sys.argv[2]
    path_to_output = sys.argv[3]

    queries = read_file(testfile)
    outputs = search(path_to_index, queries)
    if outputs:
        write_file(outputs, path_to_output)
        print(f"\nSuccessfully wrote results to {path_to_output}")


if __name__ == '__main__':
    main()
