# Machine Learning Project 2
This is the second project of the EPFL Machine Learning course, Fall 2019. In the project, we are given a dataset containing 2.5 millions tweets. Half of the tweets are labeled with positive sentiment and the rest are negative. Our task is to predict 10000 unlabeled tweets in the testing set. 

## Table of Contents

- [Dependencies](#dependencies)
- [Files Description](#files-description)
- [Result](#result)
- [Steps to Reproduce our Result](#steps-to-reproduce-our-result)
- [Developers](#developers)
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
- `bagging.ipynb`: Simple voting (could be used after training and testing in bert_based.ipynb).
- `bert_based.ipynb`: Traning and testing procedure for BERT based models
- `fasttext.ipynb`: 
- `helpers.py`: 
- `run.py`: Codes to reproduce our result (use run.sh to reproduce)
- `run.sh`: Script to reproduce our result. There are two methods to do it:  
  (1) Use trained model to get the our best prediction  
  (2) Vote from predictions to get the our best prediction (default)
  You can comment out the one you don't want. Be sure to change the *test_data_dir* argument if you choose the first method.

## Result
* AIcrowd competition link: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019
* Group name: **TWN1**
* Leaderboard 
  - **0.909** of categorical accuracy.
  - **0.909** of F1 score.

## Steps to reproduce our result
1. Change DATA_TRAIN_PATH and DATA_TEST_PATH specified in run.py to the paths of your training and testing data
2. Execute the following command
    ```bash
    $ python run.py
    ```
    
3. The prediction will be saved at OUTPUT_PATH specified in run.py (default to test.csv)


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

## License
Licensed under [MIT License](LICENSE)
