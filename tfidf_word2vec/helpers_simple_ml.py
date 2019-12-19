import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def clean_str_regex(string):
    """
    Tokenization and string cleaning for the data files
    Modify it to remove the expression such as "'m" and "'s", instead of representing them as "am" and "is".
    Remove all digits and some punctuation marks and transform all characters to lower case.

    ---> Input string :  A single line of the text file passed as a string

    ---> Output: The same string filtered with the regular expressions
    """

    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("^\d+", "", string)
    string = re.sub("\'m", "", string)
    string = re.sub("\'s", "", string)
    string = re.sub("\'ve", "", string)
    string = re.sub("n\'t", " not ", string)
    string = re.sub("\'re", "", string)
    string = re.sub("\'d", "", string)
    string = re.sub("\'ll", "", string)
    string = re.sub(",", "", string)
    string = re.sub("!", "", string)
    string = re.sub("\(", "", string)
    string = re.sub("\)", "", string)
    string = re.sub("\?", "", string)
    string = re.sub("url", "", string)
    string = re.sub("\s{2,}", " ", string)
    string = re.sub("\d*", "", string)
    return string.strip().lower()


def remove_stop_words(string, stop_words_list):
    """
    Remove all stopwords from the given string according to the list of stopwords passed as parameters.

    ---> Input string : A single line of the text file passed as a string
    ---> Input stop_words_list : The list of the stop words which must be removed from the text

    ---> Output: The same string as passed in input without the stopwords.
    """

    words = [str(w.lower()) for w in string.split() if w not in stop_words_list]
    return " ".join(words)


def clean_str(string, stop_words_list):
    """
    Perform both cleaning operations defined above :
    First, filter the string using the regular expressions.
    Second, we remove the stop words from the string.

    ---> Input string : A single line of the text file passed as a string
    ---> Input stop_words_list : The list of the stop words which must be removed from the string

    ---> Output: The initial string cleaned and ready to be used for the embeddings.
    """

    string = clean_str_regex(string)
    string = remove_stop_words(string, stop_words_list)
    return string


def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    """
    Load data from files, split the data into single line and clean each line with the cleaning functions.
    Then generate the labels (-1, or 1). To be more specific, -1 stands for negative examples while 1 represents positive examples.
    Finally, return the cleaned sentences and labels for the training set and the splitted sentences for the testing set.

    ---> Input positive_data_file : The text file containing the positive tweets
    ---> Input negative_data_file : The test file containing the negative tweets
    ---> Input test_data_file     : The test file containing the testing data

    ---> Output: A list which contains the following elements :
                 - train  : The training set which will be passed as input for the model.
                 - labels : The labels corresponding to the tweets of the training data.
                 - test   : The test set which must be passed to the trained model.
    """

    # load the data from the files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [s.strip() for s in test_examples]

    # Create the set of stopwords using the one from the nltk library
    stop_words_set = set(stopwords.words('english'))
    # Adds punctuation signs, some special characters and frequent terms
    stop_words_set.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','@', '<', '>', '-',
                           '``', '--', '—', '&', '%', '*', '•', '#', "''", 'user', 'url', 'u'])

    # Create the training set and clean both datasets
    train = positive_examples + negative_examples
    train = [clean_str(sent, stop_words_set) for sent in train]
    test = [clean_str(sent, stop_words_set) for sent in test_examples]

    # Generate the labels for the training set
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [-1 for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return [train, labels, test]


def create_submission_file(test_predictions, filename):
    """
    Create a submission csv file whose format is same as the sample submission.
    
    ---> Input test_predictions: Classification prediction from testing data.
    ---> Input filename: The file name of submission csv file
    """
    submission = pd.DataFrame({'Id': list(range(1, len(test_predictions)+1)), 'Prediction': test_predictions })
    submission.to_csv(filename, index=False)


def normalization(tweet_list):
    """
    Do lexicon normalization to recover original word meaning.
    
    ---> Input tweet_list: A list of words that are from a tweet sentence.
    ---> Output normalized_tweet: A list of normalized tweet words.
    """
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)
    return ' '.join(normalized_tweet)


def normalization_word2vec(tweet_list):
    """
    Do lexicon normalization for word2vec models to recover original word meaning.

    ---> Input tweet_list: A list of words that are from a tweet sentence.
    ---> Output normalized_tweet: A list of normalized tweet words.
    """
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet



