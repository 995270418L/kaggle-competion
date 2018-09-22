import load
import numpy as np

class CrossEntropy(object):

    @staticmethod
    def fn(a, y):
        return y * np.log(a) + (1 - y) * np.log(1-a)

    @staticmethod
    def cost_derivation(a, y):
        return a - y

class Network2(object):
    # 784, 30, 10
    def __init__(self,size):
        self.layer = len(size)
        self.bias = [np.random.randn(i, 1) for i in size[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(x) for y, x in zip(size[:-1], size[1:])]
        self.cost = CrossEntropy

    def SDG(self, train_data, epochs, min_batch_size, eta,
            lamda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):
        if evaluation_data: n_data = len(evaluation_data)
        n = len(train_data)
        evaluation_cost, evaluation_accuracy = [], []
        train_cost, train_accuracy = [], []
        for j in range(epochs):
            batch_sizes = [ train_data[ k:k+min_batch_size ] for k in range(0,n,min_batch_size)]
            for batch_size in batch_sizes:
                self.update_min_batch(batch_size,eta,lamda,n)

    def total_cost(self, data, n, convert=False):
        cost = 0.0

        cost =

    def update_min_batch(self, value, eta, lamda, n):
        nable_b = [np.zeros(b.shape) for b in self.bias]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in value:
            delta_w, delta_b = self.backprop(x, y)
            nable_w = [ nw + dw for nw,dw in zip(nable_w,delta_w)]
            nable_b = [ nb + db for nb,db in zip(nable_b,delta_b)]
        self.weights = [ (1 - eta*lamda/n)*w - (eta/len(value))*nw for w,nw in zip(self.weights,nable_w)]
        self.bias = [ b - (eta/len(value))*nb for b, nb in zip(self.bias, nable_b)]

    def backprop(self, x, y):
        delta_w = [ np.zeros(w.shape) for w in self.weights ]
        delta_b = [ np.zeros(b.shape) for b in self.bias ]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.bias,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost.cost_derivation(activations[-1], y)
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for layer in range(2, self.layer):
            sp = sigmoid_prime(zs[-layer])
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            delta_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
            delta_b[-layer] = delta
        return delta_w,delta_b


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    train_data,validation_data,test_data = load.load_data_wrapper()
    # train_data,test_data = bload.load_data_wrapper()
    # compileNp(train, train_data)
    net = Network([784, 30, 10])
    net.SGD(train_data, 30, 10, 0.5, test_data=test_data)