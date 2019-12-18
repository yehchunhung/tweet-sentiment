#!/bin/bash
# Use trained model to get the final prediction
# rm -rf models/
# git clone https://gitlab.com/epfl-machine-learning-project-2/models.git models/
# python3 run.py --test_model --test_data_dir 'data/test_data.txt'

# Vote from predictions to get the final prediction
python3 run.py --test_predictions
