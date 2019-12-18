# Machine Learning Project 2
This is the second projects of the course, Machine Learning, in EPFL. Given the Twitter dataset, we analize the classification and predict the sentiments by the texts of tweets.

## Table of Contents

- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Files Description](#files-description)
- [Result](#result)
- [Steps to Reproduce our Result](#steps-to-reproduce-our-result)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Cross Validation](#cross-validation)
- [Developers](#developers)
- [License](#license)

## Getting Started


## Dependencies

To run the project you will need the following dependencies installed:

* [FastText]
    ```sh
    $ pip install fasttext
    ```
    
* [Torchtext]
    ```sh
    $ pip install torchtext
    ```
    
* [Transformers]
    ```sh
    $ pip install transformers
    ```

## Files Description

- `activations.py`: Activation functions we used.
- `proj1_helper.py`: 
- `project1.ipynb`:
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
    python run.py
    ```
    
3. The prediction will be saved at OUTPUT_PATH specified in run.py (default to test.csv)

---


## Developers
[@Kuan Tung](https://www.aicrowd.com/participants/kuan)
[@Chun-Hung Yeh](https://www.aicrowd.com/participants/yeh)
[@De-Ling Liu](https://www.aicrowd.com/participants/snoopy)

[FastText]: <https://pypi.python.org/pypi/fasttext>
[Torchtext]: <https://pypi.org/project/torchtext/>
[Transformers]: <https://pypi.org/project/transformers/>

## License
Licensed under [MIT License](LICENSE)
