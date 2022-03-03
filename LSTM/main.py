from ast import arg
from json import load
from re import T
import numpy as np
import os
import pandas

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from src import Preprocessing
from src import RedditAnalyzer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import parameter_parser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
y_dim = 3


class DatasetMaper(Dataset):
    '''
    Handles batches of dataset
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # tmpy = [0 for i in range(3)]
        # tmpy[self.y[idx]+1] = 1
        # tmpy = torch.tensor(tmpy)
        # return self.x[idx], tmpy
        return self.x[idx], torch.Tensor(self.y[idx])


class Execute:
    '''
    Class for execution. Initializes the preprocessing as well as the 
    Tweet Classifier model
    '''

    def __init__(self, args):
        self.__init_data__(args)

        self.args = args
        self.batch_size = args.batch_size

        self.model = RedditAnalyzer(args).to(device)

    def __init_data__(self, args):
        '''
        Initialize preprocessing from raw dataset to dataset split into training and testing
        Training and test datasets are index strings that refer to tokens
        '''
        self.preprocessing = Preprocessing(args)
        self.preprocessing.load_data()
        self.preprocessing.prepare_tokens()

        raw_x_train = self.preprocessing.x_train
        raw_x_test = self.preprocessing.x_test

        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test

        self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
        self.x_test = self.preprocessing.sequence_to_token(raw_x_test)

        self.z = self.preprocessing.z

    def train(self, savePath=None):
        training_set = DatasetMaper(self.x_train, self.y_train)
        test_set = DatasetMaper(self.x_test, self.y_test)

        self.loader_training = DataLoader(
            training_set, batch_size=self.batch_size)
        self.loader_test = DataLoader(test_set)

        optimizer = optim.RMSprop(
            self.model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):

            predictions = []

            self.model.train()

            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.LongTensor).to(device)
                y = y_batch.type(torch.FloatTensor).to(device)
                y = y.view(-1, y_dim)

                y_pred = self.model(x)

                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                predictions += list(y_pred.squeeze().detach().cpu().numpy())

            test_predictions = self.evaluation()

            train_accuary = self.calculate_accuray(self.y_train, predictions)
            test_accuracy = self.calculate_accuray(
                self.y_test, test_predictions)

            print(
                f"Epoch {epoch+1}:\t loss: {loss.item():.6f},\t Train error: {train_accuary},\t Test error:{test_accuracy}")
        if(savePath):
            torch.save(self.model, os.path.join(args.path, savePath))

    def evaluation(self):

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor).to(device)
                # y = y_batch.type(torch.FloatTensor).to(device)

                y_pred = self.model(x)
                predictions += list(y_pred.detach().cpu().numpy())

        return predictions

    def test(self, loadPath=None):
        if(loadPath):
            self.model = torch.load(os.path.join(args.path, loadPath))
        self.model.eval()
        with torch.no_grad():
            for z in self.z:
                print("")
                # z is string
                sentence_embeding = self.preprocessing.sequence_to_token([z])
                sentence_embeding = torch.Tensor(sentence_embeding)
                sentence = sentence_embeding.type(torch.LongTensor).to(device)
                rslt = self.model(sentence)
                rslt = rslt.squeeze().detach().cpu().numpy()
                print(z, rslt*2-1, sep='\n')

    def generate(self, loadPath=None, generatePath=None):
        if(loadPath):
            self.model = torch.load(os.path.join(args.path, loadPath))
        self.model.eval()
        result = []
        with torch.no_grad():
            for z in self.z:
                # z is string
                sentence_embeding = self.preprocessing.sequence_to_token([z])
                sentence_embeding = torch.Tensor(sentence_embeding)
                sentence = sentence_embeding.type(torch.LongTensor).to(device)
                rslt = self.model(sentence)
                rslt = rslt.squeeze().detach().cpu().numpy()
                rslt = rslt*2-1
                result.append(rslt)
        df = pandas.DataFrame(
            result, columns=['positive', 'rebellion', 'unity'])
        df.insert(0, 'body', self.z)
        df.to_csv(os.path.join(args.path, generatePath))

    @staticmethod
    def calculate_accuray(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0

        ans = np.array([0 for i in range(y_dim)], dtype=np.float64)
        for true, pred in zip(grand_truth, predictions):
            # idp = 0
            # for i in range(1, len(pred)):
            #     if pred[i] > pred[idp]:
            #         idp = i
            # if(true-1 == idp):
            #     true_positives += 1

            # if (pred > 0.5) and (true == 1):
            #     true_positives += 1
            # elif (pred < 0.5) and (true == 0):
            #     true_negatives += 1

            # print([2*i-1 for i in true],
            #       [2*i-1 for i in pred], sep='\t', end='\n')
            # 输出预测值与真实值

            # 慎用这一句输出！
            # print(f'{pred}\t{true}', end='\n')
            res = abs(pred-true)*2
            ans += res

        return ans / len(grand_truth)


if __name__ == "__main__":
    save_path = 'pretrained/untitled_model'
    load_path = 'pretrained/3_1-1000/GME'
    generate_path = 'data/untitled.csv'

    args = parameter_parser()
    execute = Execute(args)
    np.set_printoptions(precision=6)

    # execute.train(save_path)
    execute.test(load_path)
    # execute.generate(load_path, generate_path)
