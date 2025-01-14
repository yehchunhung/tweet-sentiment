# Tweet Sentiment Classification
This is the second project of the EPFL Machine Learning course, Fall 2019. In the project, we are given a dataset containing 2.5 millions tweets. Half of the tweets are labeled with positive sentiment and the rest are negative. Our task is to predict 10000 unlabeled tweets in the testing set. 

## Table of Contents
- [Dependencies](#dependencies)
- [Files Description](#files-description)
- [Result](#result)
- [Steps to Reproduce our Result](#steps-to-reproduce-our-result)
- [Developers](#developers)
- [References](#references)
- [License](#license)

## Dependencies
We implemented in Python 3. You will need the following dependencies installed:

* [NLTK]
    ```bash
    $ pip install nltk
    ```
* [Gensim]
    ```bash
    $ pip install gensim
    ```

* [FastText]
    ```bash
    $ pip install fasttext
    ```
    
* [Torchtext]
    ```bash
    $ pip install torchtext
    ```
    
* [Transformers]
    ```bash
    $ pip install transformers
    ```

* [tqdm]
    ```bash
    $ pip install tqdm
    ```

## Files Description
- `tfidf_word2vec/tf_idf.ipynb`: Traning and testing procedure for simple ML models using TF-IDF matrx.
- `tfidf_word2vec/word2vec.ipynb`: Traning and testing procedure for simple ML models using Word2Vec matrx.
- `tfidf_word2vec/helpers_simple_ml.py`: Helpful functions used in tf_idf.ipynb and word2vec.ipynb.
- `bagging.ipynb`: Simple voting (could be used after training and testing in bert_based.ipynb).
- `bert_based.ipynb`: Traning and testing procedures for BERT based models.
- `fasttext/fasttext.ipynb`: Traning and testing procedures for fasttext based model.
- `fasttext/helpers.py`: Useful helper functions for the fasttext model
- `run.py`: Codes to reproduce our result

## Result
* AIcrowd competition link: https://www.aicrowd.com/challenges/epfl-ml-text-classification-01b777b0-a83a-412a-b6f8-f3dc53cb1bce
* Group name: **TWN1**
* Leaderboard 
  - **0.909** of categorical accuracy.
  - **0.909** of F1 score.

## Steps to reproduce our result
There are two methods to reproduce our result
1. Use trained models to get predictions and vote from them to get our best prediction. It takes about 2.5 hours to run with CPU. With GPU, it can be faster.  
2. Vote from predictions to get our best prediction. It takes only a few seconds to run.

Here are the steps:
1. If you choose the second method skip step 2. and step 3.
2. Download trained models through this Google Drive [links] and put them in a folder called `models`
3. Change the *test_data_dir* argument (the directory of testing data)
4. Execute the following command
    ```bash
    $ python3 run.py --test_model --test_data_dir 'data/test_data.txt'
    or
    $ python3 run.py --test_predictions
    ```
5. The prediction will be saved as `best_prediction.csv`


## Developers
[@Kuan Tung](https://www.aicrowd.com/participants/kuan)
[@Chun-Hung Yeh](https://www.aicrowd.com/participants/yeh)
[@De-Ling Liu](https://www.aicrowd.com/participants/snoopy)

[NLTK]: <https://pypi.org/project/nltk/>
[Gensim]: <https://pypi.org/project/gensim/>
[FastText]: <https://pypi.python.org/pypi/fasttext>
[Torchtext]: <https://pypi.org/project/torchtext/>
[Transformers]: <https://pypi.org/project/transformers/>
[tqdm]: <https://pypi.org/project/tqdm/>
[links]: <https://drive.google.com/drive/folders/18S9meEfdKjjCUAOQLQklBOjXvMF1uMw1?usp=sharing>

## References
For BERT based models:  
1. [pytorch-sentiment-analysis Tutorial 6](https://github.com/bentrevett/pytorch-sentiment-analysis)
2. [Class of transforming pandas DataFrame to torchtext Dataset](https://gist.github.com/nissan/ccb0553edb6abafd20c3dec34ee8099d)
3. [Transformers documentation](https://huggingface.co/transformers/index.html)

## License
Licensed under [MIT License](LICENSE)
