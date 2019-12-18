# Machine Learning Project 2
This is the second projects of the course, Machine Learning, in EPFL. Given the Twitter dataset, we analize the classification and predict the sentiments by the texts of tweets.

## Table of Contents

- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Files Description](#files-description)
- [Result](#result)
- [Steps to Reproduce our Result](#steps-to-reproduce-our-result)
- [Developers](#developers)
- [License](#license)

## Getting Started

This pro

In this project, we present a comprehensive study of sentiment analysis on Twitter data, where the task is to predict the smiley to be positive or negative, given the tweet message. With a fully automated framework, we developed and experimented with the most powerful proposed solutions in the related literature, including text preprocessing, text representation, also known as feature extraction, and supervised classification techniques. Different combinations of these algorithms led to a better understanding of each component and exhausting test procedures resulted in a very high classification score on our final results.

## Dependencies

To run the project you will need the following dependencies installed:

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

## Files Description

- `activations.py`: Activation functions we used.
- `proj_helper.py`: 
- `project.ipynb`:
- `run.py`: Reproduce our result (includes training and testing)

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

[FastText]: <https://pypi.python.org/pypi/fasttext>
[Torchtext]: <https://pypi.org/project/torchtext/>
[Transformers]: <https://pypi.org/project/transformers/>

## License
Licensed under [MIT License](LICENSE)
