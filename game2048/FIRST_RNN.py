"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import torch
from torch import nn
import pandas as pd
import torch
from torchvision import transforms
from PythonApplication2 import My_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20              # train the training data n times, to save time, we just train 1 epoch
batch_size = 64
TIME_STEP = 4          # rnn time step / image height
INPUT_SIZE = 4         # rnn input size / image width
LR = 0.01               # learning rated


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=256,         # rnn hidden unit
            num_layers=6,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(256, 4)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out



def main():
    print("life")
    accuracy = 0
    max_accuracy = 0
    # Mnist digital dataset
    train_data = My_dataset(
        csv_file = './train.csv',
        transform = transforms.Compose(  
            [transforms.ToTensor()]))


    # Data Loader for easy mini-batch return in training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0)

    rnn = RNN()
    if torch.cuda.is_available():
        rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):        # gives batch data
            b_x = x
            b_y = y
            if torch.cuda.is_available():
                b_x = Variable(b_x).cuda()
                b_y = Variable(b_y).cuda()
            b_x = b_x.type(torch.float)

            b_x = b_x.view(-1, 4, 4)              # reshape x to (batch, time_step, input_size)

            output = rnn(b_x)                               # rnn output
            loss = loss_func(output, b_y)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            if step % 100 == 0:
                train_output = rnn(b_x)
                train_y = torch.max(train_output, 1)[1]
                if torch.cuda.is_available():
                    train_y = Variable(train_y).cuda().data
                train_accuracy = float((train_y == b_y).sum())  / 64
                if(train_accuracy >= max_accuracy):
                    max_accuracy = train_accuracy
                if(train_accuracy == max_accuracy):
                    torch.save(rnn, 'rnn.pkl')
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % train_accuracy)
    #net2 = torch.load('rnn.pkl')

if __name__ == '__main__':
    main()