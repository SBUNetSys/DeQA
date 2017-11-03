# DrQA PrIA

This repo keeps track of changes in DrQA for PrIA project.

# Usage

- Install: 

   (0)  prepare your Python environment (like using `virtualenv`)
   
   (1) `git clone https://github.com/csarron/DrQA-pria;`
    
   (2) `cd DrQA; pip install -r requirements.txt;`

   (3) install PyTorch, see its [website](http://pytorch.org/)
   
   (4) run `./download.sh` to download all required data 
   (**warning!** needs to download **7GB** data blobs and requires **16GB** disk storage);
   
   Note that download data size is similar to the official (~7GB), but the expanded data size is smaller 
   (16GB compared to 25GB).
   This is because we save and compress the retrieval model matrix using its sparse format 
   (see function `save_sparse_csr` and `load_sparse_csr` in script `drqa/retriever/utils.py`).
    
- Run:
    
    Run `python scripts/pipeline/interactive.py` to drop into an interactive session. 
    For each question, the top span and the Wikipedia paragraph it came from are returned.

    ```
    >>> process('What is the answer to life, the universe, and everything?')
    
    Top Predictions:
    +------+--------+---------------------------------------------------+--------------+-----------+
    | Rank | Answer |                        Doc                        | Answer Score | Doc Score |
    +------+--------+---------------------------------------------------+--------------+-----------+
    |  1   |   42   | Phrases from The Hitchhiker's Guide to the Galaxy |    47242     |   141.26  |
    +------+--------+---------------------------------------------------+--------------+-----------+
    
    Contexts:
    [ Doc = Phrases from The Hitchhiker's Guide to the Galaxy ]
    The number 42 and the phrase, "Life, the universe, and everything" have
    attained cult status on the Internet. "Life, the universe, and everything" is
    a common name for the off-topic section of an Internet forum and the phrase is
    invoked in similar ways to mean "anything at all". Many chatbots, when asked
    about the meaning of life, will answer "42". Several online calculators are
    also programmed with the Question. Google Calculator will give the result to
    "the answer to life the universe and everything" as 42, as will Wolfram's
    Computational Knowledge Engine. Similarly, DuckDuckGo also gives the result of
    "the answer to the ultimate question of life, the universe and everything" as
    42. In the online community Second Life, there is a section on a sim called
    43. "42nd Life." It is devoted to this concept in the book series, and several
    attempts at recreating Milliways, the Restaurant at the End of the Universe, were made.
    ```
    
- For more details, see [Original Facebook Research DrQA](https://github.com/facebookresearch/DrQA/blob/f1105bdb57d372706d84101bd9123419a65b6961/README.md)

