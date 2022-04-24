import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num_samples = 60000 / np.power(2, np.arange(5))
    train_loss = [0.0094, 0.0092, 0.0096, 0.0117, 0.0165]
    test_loss  = [0.0202, 0.0279, 0.0383, 0.0592, 0.0813]

    plt.loglog(num_samples, train_loss, label="Train")
    plt.loglog(num_samples, test_loss, label="Test")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Loss")
    plt.title("Network Performance as Function of Training Samples")
    plt.show()
