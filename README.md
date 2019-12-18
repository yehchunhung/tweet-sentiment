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
