import random
import numpy as np
import sys

from NNModel import NNModel

random.seed(3)


def load_X(path):
    train_x = np.loadtxt(path)
    train_x = np.array(train_x).astype(np.float64)
    norm = train_x / 255.0
    return norm


def load_Y(path):
    train_y = np.loadtxt(path)
    train_y = np.array(train_y).astype(np.int)
    one_hot = []

    for y in train_y:
        zero = [0] * 10
        zero[y] = 1
        one_hot.append(zero)

    return np.array(one_hot)


def load_train_data(train_x_path, train_y_path):
    train_x = load_X(train_x_path)
    train_y = load_Y(train_y_path)

    # Shuffle
    zipped = list(zip(train_x, train_y))
    random.shuffle(zipped)
    train_x, train_y = zip(*zipped)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # Add bias
    bias = np.ones((train_x.shape[0], 1))
    train_x = np.concatenate((train_x, bias), axis=1)

    return train_x, train_y


def load_test_data(path):
    test_x = load_X(path)
    # Add bias
    bias = np.ones((test_x.shape[0], 1))
    test = np.concatenate((test_x, bias), axis=1)
    return test


def load_data(train_x_path, train_y_path, test_percentage=0.2):
    x = load_X(train_x_path)
    y = load_Y(train_y_path)

    # Shuffle
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    x, y = zip(*zipped)

    x = np.array(x)
    y = np.array(y)

    # Add bias
    bias = np.ones((x.shape[0], 1))
    x = np.concatenate((x, bias), axis=1)

    thresh = int(len(x) * (1 - test_percentage))
    train_x = x[:thresh, :]
    train_y = y[:thresh]

    test_x = x[thresh:, :]
    test_y = y[thresh:]

    return train_x, train_y, test_x, test_y


# def main():
#     if len(sys.argv) != 4:
#         print("Invalid number of arguments")
#
#     train_x_path = sys.argv[1]
#     train_y_path = sys.argv[2]
#     test_x_path = sys.argv[3]
#
#     train_x, train_y = load_data(train_x_path, train_y_path)
#
#     model = NNModel()
#
#     num_of_epochs = 4
#
#     for epoch in range(num_of_epochs):
#         loss_sum = 0.0
#         correct_predictions = 0
#
#         for i, x in enumerate(train_x):
#             y = train_y[i]
#             param = model.forward(x)
#             loss, grads = model.backward(x, y, param)
#
#             # Check if predicted correctly
#             y_hat = param["y_hat"]
#             y_pred = np.argmax(y_hat)
#             y_true = np.argmax(y)
#             correct_predictions += (y_pred == y_true)
#
#             # Add loss and gradients to sum
#             loss_sum += loss
#
#             # Update the weights
#             model.w1 = model.w1 - grads["w1"] * model.learning_rate
#             model.w2 = model.w2 - grads["w2"] * model.learning_rate
#
#         print("Epoch number: %d" % (epoch + 1))
#         print("Loss: %f" % (loss_sum / len(train_x)))
#         print("Accuracy: %f" % (correct_predictions / len(train_x)))
#
def train_and_test():
    if len(sys.argv) != 4:
        print("Invalid number of arguments")

    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]

    # train_x, train_y = load_train_data(train_x_path, train_y_path)
    train_x, train_y, test_x, test_y = load_data(train_x_path, train_y_path)
    model = NNModel()

    model.train(train_x=train_x,
                train_y=train_y,
                batch_size=20,
                num_of_epochs=70,
                lr_decay=0.0001)

    model.test(test_x=test_x,
               test_y=test_y)


def main():
    if len(sys.argv) != 4:
        print("Invalid number of arguments")

    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]

    train_x, train_y = load_train_data(train_x_path, train_y_path)
    model = NNModel()

    model.train(train_x=train_x,
                train_y=train_y,
                batch_size=20,
                num_of_epochs=75,
                lr_decay=0.00015)

    test_x = load_test_data(test_x_path)

    test_y = []

    for x in test_x:
        param = model.forward(x, False)
        y_hat = param['y_hat']
        y = np.argmax(y_hat)
        test_y.append(str(y))

    with open("test_y", "w+", encoding="utf-8") as f:
        f.write("\n".join(test_y))


main()
