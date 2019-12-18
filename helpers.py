# -*- coding: utf-8 -*-

import numpy as np
import csv

def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval:(k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)


def create_csv_submission(ids, y_pred, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
def read_file(path1, path2):
    data_pos = pd.read_table(path1, header = None)
    data_pos.columns = ['text']
    data_pos = data_pos[-data_pos['text'].duplicated()]
    data_pos['rank'] = 'positive'

    data_neg = pd.read_table(path2, header = None, error_bad_lines=False)
    data_neg.columns = ['text']
    data_neg['rank'] = 'negative'
    data_neg = data_neg[-data_neg['text'].duplicated()]

    data = pd.concat([data_pos,data_neg], ignore_index=True)
    return data


def gen_raw_data(data):
    # adjust the data into the form that fasttext needs 
    col = ['rank', 'text']
    data = data[col]
    data['rank'] = ['__label__'+ s for s in data['rank']]
    data['text'] = data['text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
    return data


def gen_pre(data):
    prediction = []
    for i in range(np.shape(data)[0]):
        pred = np.where(model.predict(data.iloc[i]['text'])[0][0] == '__label__positive',1 , -1) 
        prediction.append(pred)
    return prediction


def gen_sub():
    # generate prediction
    prediction = gen_pre(data_test)
    
    # adjust format
    data_test['Id'] = np.arange(np.shape(data_test)[0]) + 1
    data_test['Prediction'] = prediction
    create_csv_submission(data_test['Id'], data_test['Prediction'], 'predictions.csv')



def cv(d, data):
    data_cv = data
    indices = build_k_indices(data_cv['rank'], d, seed=1)
    lr = [0.01, 0.05, 0.1]
    epoch = [5, 15, 30]
    wordNgrams = [1, 2, 3]
    ws = [4, 5, 6]
    dim = [50, 100, 200]
    minCount = [1, 2, 3]
    Acc = []
    
    for o in range(len(minCount)):
        for n in range(len(dim)):
            for m in range(len(ws)):
                for l in range(len(wordNgrams)):
                    for k in range(len(epoch)):
                        for j in range(len(lr)):
                            acc = []    
                            for i in range(d):
                                data_train = data_cv.drop(indices[i])
                                data_test = data_cv.iloc[indices[i]]
                                data_train.to_csv('file/data_train.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
                                data_test.to_csv('file/data_test.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
                                model = fasttext.train_supervised(input="file/data_train.txt",
                                                                      lr = lr[j],
                                                                      epoch = epoch[k],
                                                                      wordNgrams = wordNgrams[l],
                                                                      ws = ws[m],
                                                                      dim = dim[n],
                                                                      minCount = minCount[o],
                                                                      loss = "softmax")
                    
                                acc.append(model.test("file/data_test.txt")[1])
                            print("acc={A}, lr={l}, epoch={E}, wordNgrams={N}, ws={W}, dim={D}, minCount={C}, loss={L)".format(A=np.mean(acc),l=lr[j],E=epoch[k],N=wordNgrams[l],W=ws[m],D=dim[n],C=minCount[o],L = "softmax"))
                            Acc.append(np.mean(acc))
    return Acc