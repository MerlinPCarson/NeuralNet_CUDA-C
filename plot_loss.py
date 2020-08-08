import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(history):

    plt.plot(history[0], label='Training')
    plt.plot(history[1], label='Validation')

    plt.title('Training Curves')
    plt.legend()
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.tight_layout()

    plt.show() 


def main():
    if len(sys.argv) == 2:
        history_file = sys.argv[1]
    else:
        print("Usage: python plot_loss.py <history csv file> ")
        return 1

    if not os.path.isfile(history_file):
        print(f"{history_file} not found. ")
        return 1

    history = np.genfromtxt(history_file, delimiter=',')

    plot_loss(history)

    return 0

if __name__ == "__main__":
    sys.exit(main())
