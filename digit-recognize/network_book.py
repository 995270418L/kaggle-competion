import json
import random
from time import time

import numpy as np

import bload
import load

# 383s train_data accuracy: 97% validation_data accuracy: 96% yita=0.5, lmbda=5.0
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def cost_derivation(a, y):
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes):
        self.layer = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=CrossEntropyCost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        total_cost = time()
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            start = time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {},Rating:{}".format(accuracy, n,accuracy/n*100))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, validate=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {},Rating:{}".format(accuracy, n_data,accuracy/n_data*100))
            print("one's cost time:{}".format(time() - start))
        print("all cost time:{}".format(time() - total_cost))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lamda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lamda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
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
        delta = self.cost.cost_derivation(activations[-1], y)
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for layer in range(2, self.layer):
            sp = sigmoid_prime(zs[-layer])
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (delta_w, delta_b)

    # convert: y参数是否扩展过.
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    # 是否是验证集
    def total_cost(self, data, lmbda, validate=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if validate: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        with open('weights.json', 'wt', encoding='utf-8') as f:
            json.dump(data['weights'], f)
        with open('biases.json', 'wt', encoding='utf-8') as f:
            json.dump(data['biases'], f)

#### Loading a Network
# def load(filename):
#     f = open(filename, "r")
#     data = json.load(f)
#     f.close()
#     cost = getattr(sys.modules[__name__], data["cost"])
#     net = Network(data["sizes"], cost=cost)
#     net.weights = [np.array(w) for w in data["weights"]]
#     net.biases = [np.array(b) for b in data["biases"]]
#     return net

#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    train_data,v, test_data = bload.load_data_wrapper()
    net = Network([784,30,10])
    net.SGD(train_data,30,10,0.5,5.0,test_data,True,True,True,True)