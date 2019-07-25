import matplotlib.pyplot as plt
import numpy as np

def loss_over_epochs(training_loss, evaluation_loss = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom = 0)
    #training data
    x = np.arange(1, len(training_loss)+1)
    y = training_loss
    training_line = plt.plot(x, y)
    #evaluation data
    if evaluation_loss is not None:
        x - np.arange(1, len(evaluation_loss)+1)
        y = evaluation_loss
    plt.show()

def compare_losses(losses, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom = 0, top = 0.6)
    #training data
    for loss, label in zip(losses, labels):
        x = np.arange(1, len(loss)+1)
        y = loss
        training_line = plt.plot(x, y, label=label)
    plt.xticks([i for i in range(len(losses[0]) + 1)])
    plt.title("Loss compared by Learning Rates (h)")
    plt.legend()
    plt.show()

def compare_accuracies(accs, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom = 0, top = 100)
    #training data
    for acc, label in zip(accs, labels):
        x = np.arange(1, len(acc)+1)
        y = acc
        training_line = plt.plot(x, y, label=label)
    plt.xticks([i for i in range(len(accs[0]) + 1)])
    plt.title("Accuracy compared by Learning Rates (h)")
    plt.legend()
    plt.show()
