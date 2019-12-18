# Machine Learning Project 2
This is one of the projects of the course, Machine Learning, in EPFL. We intend to analyze the data and make prediction from Higgs Boson dataset given by CERN.

## Table of Contents

- [Getting Started](#getting-started)
- [Files Description](#files-description)
- [Result](#result)
- [Steps to Reproduce our Result](#steps-to-reproduce-our-result)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Cross Validation](#cross-validation)
- [Developers](#developers)
- [License](#license)

## Getting Started

In this project, we use the concepts we have seen in the lectures and labs to this real-world dataset. Specifcially, we do  exploratory data analysis to understand the data, do feature processing and cleaning the dataset to extract more meaningful information. Finally, we implement machine learning algorithms, mainly regressions,  on real data to generate predictions to unseen data.

## Files Description

- ridge-regression-with-fine-tuning: One of our improved ML method.
- `activations.py`: Activation functions we used.
- `implementations.py`: Implementations of the six required ML methods.
- `nn.py`: Implementation of neural network.
- `proj1_helper.py`: Useful helper functions for the project (also includes one of our improved ML method (Regularized Logistic Regression with Preprocessing)).
- `project1.ipynb`: Training and testing of the six required ML methods and our improved ML methods (Regularized logistic regression and Neural Network)
- `run.py`: Reproduce our result (includes training and testing)

## Result
* AIcrowd competition link: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019
* Group name: **TWN1**
* Leaderboard 
  - **0.839** of categorical accuracy.
  - **0.754** of F1 score.

## Steps to reproduce our result
1. Change DATA_TRAIN_PATH and DATA_TEST_PATH specified in run.py to the paths of your training and testing data
2. Execute the following command
```bash
python run.py
```
3. The prediction will be saved at OUTPUT_PATH specified in run.py (default to test.csv)

---

## The following description is for the method in ridge-regression-with-fine-tuning

### Data Preparation
In the data preparation step, we transform raw data into a relatively explainable data. In details, we mainly deal with missing value and select proper features to enhance the performance. For the basic ML methods, compared with the case removing of missing data, we oberve that missing value can still make contribution to accuracy. As a result, we keep those features with missing value for training. However, in our improved methods, we transform original data. For instance, we replace missing value with the median from the distribution of that certain feature. Additionally, we group the dataset by the feature, PRI_jet_num, to eliminate the missing value resulted from the physical constraints.


### Feature Engineering
Besides replacing missing value to some features, we augment features based on our physics knowledge as well. That is, we assume there could be some complex theoretical relations between these physical quantities. Therefore, we not only expand each  feature by the polynomial basis, but also add cross terms to catch the significance of different features. With some features related to angles, we also apply sine and cosine functions to create more complex features.


### Cross Validation
For choosing our best model from our improved methods, we exert 10-fold cross validation to help us tune hyperparameters. Specifically, we design a range of values as possible hyperparameters to polynomial degrees and regularization coefficients respectively. After calculation, we select the model that has the highest accuracy average as our ultimate model to submit.

## Developers
[@Kuan Tung](https://www.aicrowd.com/participants/kuan)
[@Chun-Hung Yeh](https://www.aicrowd.com/participants/yeh)
[@De-Ling Liu](https://www.aicrowd.com/participants/snoopy)

## License
Licensed under [MIT License](LICENSE)
