class NeuralNetwork:

    def __init__(self, inputNodes, hiddennodes, outputnodes, learningrate):
        self.inode = inputNodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        pass

    def train(self):
        pass


    def query(self):
        pass

if __name__ == "__main__":
    print("hello world")
    nn = NeuralNetwork(3,3,3,3)