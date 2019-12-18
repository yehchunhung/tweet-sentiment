#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to reproduce the exact same prediction which we used in our best submission to the competition on AIcrowd
"""

from argparse import ArgumentParser
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from torchtext import data
from transformers import AutoTokenizer

SEED = 77

# ensure reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            label = row.label
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    @classmethod
    def splits(cls, text_field, label_field, test_df=None, **kwargs):
        test_data = cls(test_df.copy(), text_field, label_field, is_test=True, **kwargs)
        return test_data


def test(model, iterator):
    """ Testing procedure"""
    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator):
            _, logits = model(batch.text, labels=batch.label)[:2]
            softmax = torch.softmax(logits, dim=1)
            final_preds = torch.max(softmax, 1, keepdim=True)[1].squeeze(1)
            predictions.extend(final_preds.tolist())

    return predictions


def test_model(test_data_dir):
    """ Use trained models to get the final prediction """
    pretrained_models = ['bert-base-uncased', 'xlnet-base-cased', 'roberta-base']
    for pretrained_model in pretrained_models:
        # load model
        if pretrained_model == 'bert-base-uncased':
            from transformers import BertForSequenceClassification as SequenceClassificationModel
            selected_epochs = [1, 2, 3, 4, 5]
        elif pretrained_model == 'xlnet-base-cased':
            from transformers import XLNetForSequenceClassification as SequenceClassificationModel
            selected_epochs = [1, 2, 3]
        elif pretrained_model == 'roberta-base':
            from transformers import RobertaForSequenceClassification as SequenceClassificationModel
            selected_epochs = [1, 2, 3, 4, 5]

        model = SequenceClassificationModel.from_pretrained(pretrained_model)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id

        max_input_length = tokenizer.max_model_input_sizes[pretrained_model]

        def tokenize_and_cut(sentence):
            """ Tokenize the sentence and cut it if it's too long """
            tokens = tokenizer.tokenize(sentence)
            # - 2 is for cls and sep tokens
            tokens = tokens[:max_input_length - 2]
            return tokens

        # xlnet model has no max_model_input_sizes field but it acutally has a limit
        # so we manually set it
        if max_input_length == None:
            max_input_length = 512

        # Field handles the conversion to Tensor (tokenizing)
        TEXT = data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=tokenize_and_cut,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=init_token_idx,
            eos_token=eos_token_idx,
            pad_token=pad_token_idx,
            unk_token=unk_token_idx
        )

        LABEL = data.LabelField(dtype=torch.long, use_vocab=False)

        with open(test_data_dir) as f:
            test_lines = [line.rstrip('\n')[line.rstrip('\n').find(',') + 1:] for line in f]

        test_df = pd.DataFrame(test_lines, columns=['text'])
        # because the model input required some label
        # we won't use this though
        test_df['label'] = 1

        # transform DataFrame into torchtext Dataset
        print('Transforming testing data for', pretrained_model, 'model')
        test_data = DataFrameDataset.splits(text_field=TEXT, label_field=LABEL, test_df=test_df)

        BATCH_SIZE = 32
        # get gpu if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_iterator = data.Iterator(test_data, batch_size=BATCH_SIZE, device=device, shuffle=False, sort=False, train=False)

        for selected_epoch in selected_epochs:
            # load trained model
            model.load_state_dict(
                torch.load(os.path.join(
                    'models',
                    f'{pretrained_model}-e{selected_epoch:02}-model.pt'
                ), map_location=device)
            )
            model = model.eval()

            # get predictions of test data
            print(f'Testing for {pretrained_model} epoch {selected_epoch}')
            predictions = test(model, test_iterator)

            # map predictions to match the original
            label_map = {0: -1, 1: 1}
            corrected_predictions = list(map(lambda x: label_map[x], predictions))

            # load data into dataframe
            submission = pd.read_csv('predictions_test/sample_submission.csv')
            submission.Prediction = corrected_predictions
            submission.to_csv(os.path.join('predictions_test', f'{pretrained_model}-e{selected_epoch:02}.csv'), index=False)

    test_predictions('predictions_test')


def test_predictions(base_dir='predictions'):
    """ Vote from predictions to get the final prediction """
    bert = 'bert-base-uncased'
    roberta = 'roberta-base'
    xlnet = 'xlnet-base-cased'

    bert_picks = [1, 2, 3, 4, 5]
    roberta_picks = [1, 2, 3, 4, 5]
    xlnet_picks = [1, 2, 3]

    # load all predictions
    predictions = []
    for i in bert_picks:
        predictions.append(pd.read_csv(os.path.join(base_dir, f'{bert}-e{i:02}.csv')))
    for i in roberta_picks:
        predictions.append(pd.read_csv(os.path.join(base_dir, f'{roberta}-e{i:02}.csv')))
    for i in xlnet_picks:
        predictions.append(pd.read_csv(os.path.join(base_dir, f'{xlnet}-e{i:02}.csv')))

    total_length = len(predictions)

    submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    num_preds = submission.shape[0]

    bagging_predictions = np.zeros(num_preds, dtype=int)

    for i in range(num_preds):
        votes = {-1: 0, 1: 0}
        for j in range(total_length):
            votes[predictions[j].Prediction[i]] += 1
        # pick the label with the higher votes
        bagging_predictions[i] = max(votes, key=votes.get)

    submission.Prediction = bagging_predictions
    submission.to_csv('best_prediction.csv', index=False)


def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--test_model', action='store_true',
                        help='Use trained model to get the final prediction')
    parser.add_argument('--test_predictions', action='store_true',
                        help='Vote from predictions to get the final prediction')
    parser.add_argument('--test_data_dir', type=str, help='Directory of testing data')
    args = parser.parse_args()

    if args.test_model:
        test_model(test_data_dir=args.test_data_dir)
    elif args.test_predictions:
        test_predictions()


if __name__ == '__main__':

    main()
