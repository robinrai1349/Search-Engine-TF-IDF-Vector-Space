# Information Retrieval System README

## Overview

This project has been pursued to create a Python-based information retrieval system. The system is designed to efficiently retrieve relevant documents from a collection of HTML files related to video games.

## System Components

The system is split into two main components:

- **index.py**: Responsible for building the Inverted Index Model by processing the raw text of HTML documents.
- **search.py**: Handles the retrieval process using the Vector Space Model, including preprocessing, vectorisation, and ranking

## System Requirements

- Python 3.x
- Required Python libraries (install using `pip install [library_name]`):
    - NLTK (Natural Language Toolkit)
    - FuzzyWuzzy

## Usage

### System Customization

The preprocessing techniques and search functionalities within the system are customisable. However, the settings the files come with have been experimented to be the most effective and efficient state of the system.

To customise, open the respective files (`index.py` and `search.py`) and toggle the options based on your preferences.

### Running the System

1. **Install Dependecies:**
   - Run the following commands to install the required libraries:
   ```
   pip install nltk
   pip install fuzzywuzzy
   ```

2. **Run Indexing:**
   - Execute the following command in the terminal to build the Inverted Index Model:
   ```
   python index.py
   ```
   
3. **Run Search:**
   - Start the search functionality by running the following command:
   ```
   python search.py
   ```
   Follow the prompt to begin searching.
   To exit, type in 'quit' and hit `ENTER`.

4. **Customisation:**
   - Modify the system's behaviour by toggling and adjusting preprocessing techniques and search functionalities in the code

## System Evaluation

Refer to the provided flyer and presentation for a detailed analysis of the system's performance and experiements conducted.