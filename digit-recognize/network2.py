import json
import random
from time import time

import bload
import load2
import numpy as np

class CrossEntropy(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1-a)))

    @staticmethod
    def cost_derivation(a, y):
        return a - y

class Network2(object):
    # 784, 30, 10
    def __init__(self,size):
        self.layer = len(size)
        self.biases = [np.random.randn(i, 1) for i in size[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x,y in zip(size[:-1], size[1:])]
        self.cost = CrossEntropy
        self.sizes = size

    def SDG(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):
        total_time = time()
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        train_cost, train_accuracy = [], []
        for j in range(epochs):
            start = time()
            random.shuffle(train_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            if monitor_training_cost:
                cost = self.total_cost(train_data,lmbda)
                train_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(train_data, True)
                train_accuracy.append(accuracy)
                print( "Accuracy on training data: {} / {}, Rating:{}".format(accuracy, n, accuracy/n * 100 ))
            if monitor_evaluation_cost:
                cost = self.total_cost(validation_data, lmbda, True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(validation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}, Rating:{}".format(accuracy, n_data, accuracy/n_data * 100))
            print("cost_time:{}".format(time() - start))
        print("total_cost_time:{}".format(time() - total_time))
        return evaluation_cost, evaluation_accuracy, train_cost, train_accuracy

    # convert 变量 y 表示是否被正则化
    def total_cost(self, data, lmbda, validate=False):
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            # 如果不是训练数据
            if validate:
                y = self.vector_result(y)
            cost += self.cost.fn(a, y) / len(data)
        # np.linalg.norm 表示的是对w矩阵求二次范数, 即 sqrt(x1**2 + x2**2 + x3**2 +...+ xn**2)
        cost += lmbda / (2 * len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def vector_result(self, y):
        e = np.zeros((10,1))
        e[y] = 1.0
        return e

    def accuracy(self, test_data, convert=False):
        if convert:
            test_results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x, y) in test_data]
        else:
            test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(y_p == y_r) for (y_p, y_r) in test_results)

    def forward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def update_mini_batch(self, mini_batch, eta, lamda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lamda / n)) * w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = activations[-1] - y
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for layer in range(2, self.layer):
            sp = sigmoid_prime(zs[-layer])
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (delta_b, delta_w)

    def save(self):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        with open('network_weights.json', 'wt', encoding='utf-8') as f:
            json.dump(data['weights'], f)
        with open('network_biases.json', 'wt', encoding='utf-8') as f:
            json.dump(data['biases'], f)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    train_data, validation_data, test_data = load2.load_data_wrapper()
    net = Network2([784, 30, 10])
    net.SDG(train_data, 1, 10, eta=0.5, lmbda=5.0,
            evaluation_data=test_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)