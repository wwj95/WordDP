# Certified Robustness to Word Substitution Attack with Differential Privacy(WordDP)

This repository contains Pytorch implementations of the NAACL2021 paper 
[Certified Robustness to Word Substitution Attack with Differential Privacy](http://cs.emory.edu/site/aims/pub/wang21naacl.pdf).

## Overview
* `PWWS` is the directory for [PWWS attack](https://github.com/JHL-HUST/PWWS), `fool_result` and `runs` are from `PWWS`
* `certified_eps` is the directory for training data for `wordDP_training`. 
* `wordDP_training.py` is the training file for neural networks.
* `wordDP_v4_2.py` is the file for WordDP, which builds the exponential mechanism-based algorithm as the randomized mechanism
to achieve certified robustness.

## Dependencies
* Python 3.7.1.
* If you did not download WordNet(a lexical database for the English language), use `nltk.download('wordnet')` to do so.(Cancel the code comment on line 14 in `paraphrase. py`) 


## Usage

* Download dataset files from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) , which include
    - IMDB: `aclImdb.zip`. Decompression and place the folder`aclImdb` in`data_set/`.
    - AG's News: `ag_news_csv.zip`. Decompression and place the folder `ag_news_csv` in`data_set/`.
* Download `glove.6B.100d.txt`from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) and place the file in `/`.
* Run `wordDP_training.py` by using, e.g., `python3 wordDP_training.py -m word_lstm -d agnews -L 5`, you can training neural networks
`word_lstm` on dataset `agnews` with neighbour distance `L=5`.
* Run `wordDP_v4_2.py` by using, e.g., `python3 wordDP_v4_2.py -L 5 -L_adv 20 -eps 1.2 -m  word_lstm  -ita 4 --start 0 -d agnews`, you
can achieve wordDP on dataset `agnews` with neighbour distance `L=5`.


## Contact

* If you have any questions regarding the code, please create an issue or contact the [owner]() of this repository.

##  Acknowledgments

- Code refer to: [PWWS attack](https://github.com/JHL-HUST/PWWS)
