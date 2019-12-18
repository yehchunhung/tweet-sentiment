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



In this project, we use the concepts we have seen in the lectures and labs to this real-world dataset. Specifcially, we do exploratory data analysis to understand the data, do feature processing and cleaning the dataset to extract more meaningful information. Finally, we implement machine learning algorithms, mainly regressions, on real data to generate predictions to unseen data.

Sentiment analysis is an interesting problem to give machines the ability to understand human emotion. This is a challenging task due to the complexity of languages, which make use of rhetorical devices such as sarcasm or irony. Twitter is a popular social medium for people to convey opinions; hence a successful sentiment classifier based on its data could offer interesting trends regarding prominent topics in the news. For example, one could gauge the popular opinion of politicians by calculating the sentiment from all tweets containing the politicians' information. Sentiment analysis in Twitter is a significantly different paradigm since its users are only allowed to post short tweets. Moreover, from most social network platform, users create their own words and spelling shortcuts making this task even more challenging. The aim of this project is to build an accurate text classifier for tweets. That is, our classification model determines whether the given tweet reflects positivity or negativity on the users' behalf. In addition, we would like to discuss numerous methods for building such classifier including classic ML algorithms and advanced neural networks.

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
