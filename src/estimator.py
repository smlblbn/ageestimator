import torch
import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(483)

x_train_np = np.load('../data/train.npy')
y_train_np = np.load('../data/train_gt.npy')

x_valid_np = np.load('../data/valid.npy')
y_valid_np = np.load('../data/valid_gt.npy')

x_test_np = np.load('../data/test.npy')

x_train = Variable(torch.from_numpy(x_train_np))
y_train = Variable(torch.from_numpy(y_train_np.reshape(y_train_np.shape[0], 1)).float())

x_valid = Variable(torch.from_numpy(x_valid_np))
y_valid = Variable(torch.from_numpy(y_valid_np.reshape(y_valid_np.shape[0], 1)).float())

x_test = Variable(torch.from_numpy(x_test_np))

class Net0(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net0, self).__init__()
        self.predict = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = self.predict(x)
        return x

class Net1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net1, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class Net2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden1, n_output):
        super(Net2, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
        self.predict = torch.nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x


class Net3(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden1, n_hidden2, n_output):
        super(Net3, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


net0 = Net0(n_feature=512, n_output=1)
net1 = Net1(n_feature=512, n_hidden=512, n_output=1)
net2 = Net2(n_feature=512, n_hidden=512, n_hidden1=256, n_output=1)
net3 = Net3(n_feature=512, n_hidden=512, n_hidden1=256, n_hidden2=128, n_output=1)

models = [net0, net1, net2, net3]
epochs = 500

for j in range(len(models)):
    loss_train_arr = []
    loss_valid_arr = []

    net = models[j]

    optimizer = torch.optim.RMSprop(net.parameters(), lr=2e-4)
    loss_func = torch.nn.MSELoss()

    for i in range(epochs):
        y_pred_train = net(x_train)

        loss_train = loss_func(y_pred_train, y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        y_pred_valid = net(x_valid)
        loss_valid = loss_func(y_pred_valid, y_valid)

        loss_train_arr.append(loss_train)
        loss_valid_arr.append(loss_valid)

    directory = '../estimations'
    if not os.path.exists(directory):
        os.makedirs(directory)

    y_pred_valid = net(x_valid)
    with open('../estimations/estimations_valid_net' + str(j) + '.npy', 'wb') as file:
        np.save(file, y_pred_valid.detach().numpy())

    y_pred_test = net(x_test)
    with open('../estimations/estimations_test_net' + str(j) + '.npy', 'wb') as file:
        np.save(file, y_pred_test.detach().numpy())

    directory = '../graphics'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.plot(np.arange(epochs), np.array(loss_train_arr), label='train')
    plt.plot(np.arange(epochs), np.array(loss_valid_arr), label='validation')
    plt.title('Model Loss')
    plt.legend(loc='upper right')
    plt.savefig('../graphics/loss_net' + str(j) + '.png')
    plt.show()

    threshold = np.fabs(y_pred_valid.detach().numpy() - y_valid.detach().numpy())

    min_img = np.argmin(threshold, axis=0)
    max_img = np.argmax(threshold, axis=0)
    mean_img = np.argmin(np.fabs(threshold - np.mean(threshold)), axis=0)

    print('best image:', min_img[0], ' pred age: ', y_pred_valid[min_img].data.numpy()[0][0],
          ' real age: ', y_valid[min_img].data.numpy()[0][0])
    print('moderate image: ', mean_img[0], ' pred age: ', y_pred_valid[mean_img].data.numpy()[0][0],
          ' real age: ', y_valid[mean_img].data.numpy()[0][0])
    print('worst image: ', max_img[0], ' pred age: ', y_pred_valid[max_img].data.numpy()[0][0],
          ' real age: ', y_valid[max_img].data.numpy()[0][0])
    print()

    if os.path.exists('../ismail.npy'):
        ismail = np.load('../ismail.npy')
        ismail_ = net(Variable(torch.from_numpy(ismail)))
        print('ismail pred age: ', ismail_.data.numpy()[0][0])

    if os.path.exists('../ismail_sunglasses.npy'):
        ismail = np.load('../ismail_sunglasses.npy')
        ismail_ = net(Variable(torch.from_numpy(ismail)))
        print('ismail_sunglasses pred age: ', ismail_.data.numpy()[0][0])
    print()