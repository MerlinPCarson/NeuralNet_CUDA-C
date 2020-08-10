import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data


def plot_loss(loss, val_loss):

    epochs = np.arange(1,len(loss)+1, 1)

    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')

    plt.title('Training Curves')
    plt.legend()
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.tight_layout()

    plt.show() 


def plot_acc(accs):

    epochs = np.arange(1,len(accs)+1, 1)

    plt.plot(epochs, accs, label='Test set')

    plt.title('Accuracy')
    plt.legend()
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.tight_layout()

    plt.show() 

def load_history(history_file):

    with open(history_file, 'r') as f:
        loss = [float(val) for val in f.readline().split(',')]
        val_loss = [float(val) for val in f.readline().split(',')]
        accs = [float(val) for val in f.readline().split(',')]
        preds = [int(val) for val in f.readline().split(',')]
        targets = [int(val) for val in f.readline().split(',')]

    print(loss)
    print(val_loss)
    print(accs)
    print(preds)
    print(targets)

    return loss, val_loss, accs, preds, targets

def main():
    if len(sys.argv) == 2:
        history_file = sys.argv[1]
    else:
        print("Usage: python plot_loss.py <history csv file> ")
        return 1

    if not os.path.isfile(history_file):
        print(f"{history_file} not found. ")
        return 1

    #history = np.genfromtxt(history_file, delimiter=',')
    loss, val_loss, accs, preds, targets = load_history(history_file)

    plot_loss(loss, val_loss)
    plot_acc(accs)

    print(len(np.where(np.array(preds) == np.array(targets))[0]))
    labels = [x for x in range(10)]

    plot_confusion_matrix_from_data(targets, preds, columns=labels)

    return 0

if __name__ == "__main__":
    sys.exit(main())
