import numpy as np
from sklearn import datasets

data, target = datasets.load_boston(return_X_y=True)
print('Data shape', data.shape)
print('Target shape', target.shape)
target = target.reshape(-1, 1)
print('New target shape', target.shape)


def rmse_error(pred, y):
    return np.mean((pred-y)**2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NN(object):
    def __init__(self, s, learning_rate, iterations=100):
        self.s = s
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.d = X.shape[1]
        self.w_1 = np.random.random((self.s, self.d)) * 2 - 1
        self.b_1 = np.random.random((1, self.s))
        self.w_2 = np.random.random((1, self.s)) * 2 - 1
        self.b_2 = np.random.random((1, 1))

        for i in range(self.iterations):
            # For printing
            pred = self.predict(X)
            rmse = rmse_error(pred, y)
            print('Iteration', i, ' error =', rmse)
            # For optimization
            self.optimization()

    def predict(self, X_test):
        self.z_1 = np.dot(X_test, self.w_1.T) + self.b_1
        # Sigmoid activation
        self.a_1 = sigmoid(self.z_1)

        self.z_2 = np.dot(self.a_1, self.w_2.T) + self.b_2
        # Linear activation
        self.a_2 = self.z_2

        return self.a_2

    def optimization(self):
        der_C_a_2 = self.predict(self.X) - self.y
        der_a_2_z_2 = 1
        der_z_2_w_2 = self.a_1
        der_z_2_b_2 = 1

        delta = der_C_a_2 * der_a_2_z_2
        der_C_w_2 = delta * der_z_2_w_2
        der_C_b_2 = delta * der_z_2_b_2

        w_2 = self.w_2 - self.learning_rate * np.mean(der_C_w_2, axis=0)
        b_2 = self.b_2 - self.learning_rate * np.mean(der_C_b_2, axis=0)

        der_z_2_a_1 = self.w_2  # (1, s)
        der_a_1_z_1 = self.a_1 * (1 - self.a_1)  # (n, s)
        der_z_1_w_1 = self.X  # (n, k)
        der_z_1_b_1 = 1

        der_C_w_1 = np.dot((delta * der_z_2_a_1 * der_a_1_z_1).T, der_z_1_w_1)
        der_C_b_1 = delta * der_z_2_a_1 * der_a_1_z_1 * der_z_1_b_1

        w_1 = self.w_1 - self.learning_rate * der_C_w_1
        b_1 = self.b_1 - self.learning_rate * np.mean(der_C_b_1, axis=0)

        self.w_1 = w_1
        self.b_1 = b_1
        self.w_2 = w_2
        self.b_2 = b_2


nn = NN(s=100, learning_rate=0.01, iterations=1000)
nn.fit(X=data, y=target)
rmse_error(nn.predict(data), target)
