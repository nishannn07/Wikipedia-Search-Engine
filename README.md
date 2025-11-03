# Simple Wikipedia XML Search Engine

This project is a simple, two-phase search engine for Wikipedia XML dumps. It consists of:

1.  **`phase_1_create_index.py`**: An indexer that parses a Wikipedia XML dump and builds a set of inverted index files.
2.  **`phase_1_search.py`**: A searcher that takes user queries, searches the index, and returns matching document titles.

## Features

* Parses large Wikipedia XML dumps efficiently using `iterparse`.
* Builds separate inverted indexes for four fields: **Title**, **Body**, **Category**, and **Infobox**.
* Uses NLTK's `SnowballStemmer` for word stemming.
* Supports both:
    * **Non-fielded search** (e.g., `olympic games`)
    * **Fielded search** (e.g., `t:greece c:mythology`)
* Returns the top 10 matching document titles for a query.

## Requirements

* Python 3
* NLTK: `pip install nltk`
* A Wikipedia XML dump file (e.g., `enwiki-latest-pages-articles.xml`)
* A stop-word file named `stop_words.txt` in the same directory.
    * **Important:** Words in this file must be comma-separated and individually quoted (e.g., `"a","the","is","in"`).

## How to Use

### Step 1: Create the Index

Run the `phase_1_create_index.py` script, providing the path to your Wikipedia dump and a directory to store the index files.

```bash
# Usage: python phase_1_create_index.py <path_to_wiki.xml> <path_to_index_dir>

# Example:
mkdir ./my_index
python phase_1_create_index.py /data/enwiki.xml ./my_index
```

This will create several files in the `./my_index` directory, including `title.txt`, `body_text.txt`, `word_position.pickle`, and `title_doc_no.pickle`.

### Step 2: Search the Index

1.  Create a query file (e.g., `queries.txt`) with one query per line.

    **`queries.txt` example:**
    ```
    t:greece
    olympic games
    body:zeus c:mythology
    ```

2.  Run the `phase_1_search.py` script, providing the path to your index, your query file, and an output file for the results.

```bash
# Usage: python phase_1_search.py <path_to_index_dir> <path_to_queries.txt> <path_to_output.txt>

# Example:
python phase_1_search.py ./my_index ./queries.txt ./results.txt
```

The script will print the time taken for each query and write the matching document titles to `results.txt`.
