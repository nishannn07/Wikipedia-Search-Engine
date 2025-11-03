# Simple Title Indexer

This is a Python script designed to process a Wikipedia XML file and extract information from `<title>` tags. It cleans the titles, removes common stop words, and builds an in-memory index.

## Features

* **Title Preprocessing:** Reads an XML file, finds all `<title>` tags, and tokenizes the text.
* **Text Cleaning:** Removes stop words (using NLTK) and tokens shorter than 3 characters.
* **File Output:** Creates a `title.txt` file containing one cleaned title per line.
* **In-Memory Index:**
    * `build_title_dict()`: Counts the total frequency of every word across all titles.
    * `build_posting_list()`: Creates a simple posting list for title words, recording the document number, word frequency, and file position.

## How to Use

This script is set up to run in an environment like Google Colab or a Jupyter Notebook.

1.  **Upload Files:** You must upload your Wikipedia XML file.
2.  **Install NLTK:** Make sure you have NLTK installed (`pip install nltk`).
3.  **Update Filename:** Change this line in the script to match the name of your XML file:
    ```python
    preprocess_titles("enwiki-latest-pages-articles26.xml-p42567204p42663461")
    ```
4.  **Run:** Execute the entire script. It will:
    * Download NLTK stopwords.
    * Create the `title.txt` file.
    * Create the `title_dict` and `posting_list` variables in memory.

## ⚠️ Important Limitations

This script has several major limitations and **will not work** on a full Wikipedia dump.

1.  **Fatal Memory Error:** The `preprocess_titles` function uses `f.readlines()`, which tries to load the **entire XML file into computer memory**. This will crash your computer or notebook if the file is large. It will only work on very small sample XML files.
2.  **Incomplete:** The script **only processes `<title>` tags**. The functions for processing the body text, categories, or infoboxes (`clean_text`, `build_content_dict`, etc.) are empty.
3.  **No Stemming:** The script does not perform stemming, so "run" and "running" are treated as two different words.
4.  **Simple Tokenization:** It splits titles using `title.split()`, which is not very robust and can handle punctuation or complex names incorrectly.
