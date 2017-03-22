import matplotlib.pyplot as plt
import numpy as np


def prototype():
    # create some randomly ddistributed data:
    data = np.random.randn(10000)
    print(data)
    # sort the data:
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = range(len(data))

    # plot the sorted data:

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(p, data_sorted)
    ax1.set_xlabel('$p$')
    ax1.set_ylabel('$x$')

    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')
    plt.show()

if __name__ == "__main__":
    data = np.random.randn(10000)
    print(data)
    data_sorted = np.sort(data)
    # calculate the proportional values of samples
    p = range(len(data))
    print(p)