import numpy
import numpy as np


class Deterministic:
    lb = 0
    ub = 1

    def __init__(self):
        pass

    def generate(self, x=None, nodes=10, steps=100, network=None, k_i=None, retsteps=True):
        if x is not None or network is not None:
            nodes = len(x) if x is not None else len(network)
        x = np.linspace(self.lb, self.ub, nodes) if x is None else x
        assert steps >= 0
        network = network if network is not None else self.createNetwork(nodes)
        k_i = k_i if k_i is not None else self.createK_i(network)

        values = np.zeros((nodes, steps + 1))
        values[:, 0] = x
        for i in range(steps):
            values[:, i + 1] = values[:, i] + self.step(values[:, i], network=network, k_i=k_i)
            ubCorrection = values[:, i + 1] > self.ub
            values[:, i + 1] = values[:, i + 1] * numpy.invert(ubCorrection) + ubCorrection * self.ub
            lbCorrection = values[:, i + 1] < self.lb
            values[:, i + 1] = values[:, i + 1] * numpy.invert(lbCorrection) + lbCorrection * self.lb
        return (np.linspace(0, steps, steps + 1), values) if retsteps else values

    def step(self, x, network=None, k_i=None):
        if network is None:
            network = self.createNetwork(x)
        if k_i is None:
            k_i = self.createK_i(network)
        dif = -np.subtract.outer(x, x)
        phi_ij = self.phi(self.norm(dif))
        sigma = network * phi_ij * dif
        return k_i * np.sum(sigma, axis=1)

    def phi(self, h):
        return 1 + 0 * h

    def norm(self, x):
        return np.abs(x)

    def createNetwork(self, nodes):
        return np.ones((nodes, nodes))

    def createK_i(self, network):
        nodes = len(network[0])
        no_diagonal = network * (np.diag(np.full(nodes, -1)) + np.ones((nodes, nodes)))
        horizontal_sum = np.sum(no_diagonal, axis=1)
        return 1 / (horizontal_sum + (horizontal_sum == 0)*1)


class Stochastic:
    lb = 0
    ub = 1

    def __init__(self):
        pass

    def generate(self, x=None, nodes=10, steps=100, network=None, k_i=None, retsteps=True):
        if x is not None or network is not None:
            nodes = len(x) if x is not None else len(network)
        x = np.linspace(self.lb, self.ub, nodes) if x is None else x
        assert steps >= 0
        network = network if network is not None else self.createNetwork(nodes)
        k_i = k_i if k_i is not None else self.createK_i(network)

        values = np.zeros((nodes, steps + 1))
        values[:, 0] = x
        for i in range(steps):
            values[:, i + 1] = values[:, i] + self.step(values[:, i], network=network, k_i=k_i)
            ubCorrection = values[:, i + 1] > self.ub
            values[:, i + 1] = values[:, i + 1] * numpy.invert(ubCorrection) + ubCorrection * self.ub
            lbCorrection = values[:, i + 1] < self.lb
            values[:, i + 1] = values[:, i + 1] * numpy.invert(lbCorrection) + lbCorrection * self.lb
        return (np.linspace(0, steps, steps + 1), values) if retsteps else values

    def step(self, x, network=None, k_i=None):
        if network is None:
            network = self.createNetwork(x)
        if k_i is None:
            k_i = self.createK_i(network)
        dif = -np.subtract.outer(x, x)
        phi_ij = self.phi(self.norm(dif))
        sigma = network * phi_ij * dif + self.stochastic(x)
        return k_i * np.sum(sigma, axis=1)

    def stochastic(self, x):
        return 0.1* (np.random.random(x.shape)*2-1)

    def phi(self, h):
        return 1 + 0 * h

    def norm(self, x):
        return np.abs(x)

    def createNetwork(self, nodes):
        return np.ones((nodes, nodes))

    def createK_i(self, network):
        nodes = len(network[0])
        no_diagonal = network * (np.diag(np.full(nodes, -1)) + np.ones((nodes, nodes)))
        horizontal_sum = np.sum(no_diagonal, axis=1)
        return 1 / (horizontal_sum + (horizontal_sum == 0)*1)


class MultiDim:
    lb = 0
    ub = 1

    def __init__(self):
        pass

    def generate(self, x=None, nodes=10, steps=100, network=None, k_i=None, retsteps=True):
        if x is not None or network is not None:
            nodes = len(x) if x is not None else len(network)
        x = np.linspace(self.lb, self.ub, nodes) if x is None else x
        assert steps >= 0
        network = network if network is not None else self.createNetwork(nodes)
        k_i = k_i if k_i is not None else self.createK_i(network)

        values = np.zeros((nodes, steps + 1))
        values[:, 0] = x
        for i in range(steps):
            values[:, i + 1] = values[:, i] + self.step(values[:, i], network=network, k_i=k_i)
            ubCorrection = values[:, i + 1] > self.ub
            values[:, i + 1] = values[:, i + 1] * numpy.invert(ubCorrection) + ubCorrection * self.ub
            lbCorrection = values[:, i + 1] < self.lb
            values[:, i + 1] = values[:, i + 1] * numpy.invert(lbCorrection) + lbCorrection * self.lb
        return (np.linspace(0, steps, steps + 1), values) if retsteps else values

    def step(self, x, network=None, k_i=None):
        if network is None:
            network = self.createNetwork(x)
        if k_i is None:
            k_i = self.createK_i(network)
        dif = -np.subtract.outer(x, x)
        phi_ij = self.phi(self.norm(dif))
        sigma = network * phi_ij * dif
        return k_i * (np.sum(sigma, axis=1) + self.stochastic(x))

    def stochastic(self, x):
        return 0.1* (np.random.random(x.shape)*2-1)

    def phi(self, h):
        return 1 + 0 * h

    def norm(self, x_ij):
        return np.abs(x)

    def createNetwork(self, nodes):
        return np.ones((nodes, nodes))

    def createK_i(self, network):
        nodes = len(network[0])
        no_diagonal = network * (np.diag(np.full(nodes, -1)) + np.ones((nodes, nodes)))
        horizontal_sum = np.sum(no_diagonal, axis=1)
        return 1 / (horizontal_sum + (horizontal_sum == 0)*1)

def test_func(attr1: float, attr2: int):
    """
    Function to
    :param attr1:
    :param attr2:
    :return:
    """
    print("yay")


if __name__ == "__main__":
    model = Deterministic()
    steps, nodes = model.generate()
