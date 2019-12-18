class NNModel:

    def __init__(self, dropout=0.8):
        np.random.seed(42)
        self.input_len = 28 * 28 + 1
        self.h0_len = 100
        self.out_len = 10
        self.w1 = np.random.uniform(low=-0.08, high=0.08, size=(self.input_len, self.h0_len))
        self.w2 = np.random.uniform(low=-0.08, high=0.08, size=(self.h0_len, self.out_len))

        self.learning_rate = 0.015
        self.dropout_rate = dropout

    def forward(self, x, train_mode):
        # if train_mode:
        #     ux = F.dropout(x, self.dropout_rate)
        #     x = x * ux
        # else:
        #     x *= self.dropout_rate

        z1 = np.matmul(x, self.w1)
        h0 = F.relu(z1)

        if train_mode:
            u0 = F.dropout(h0, self.dropout_rate)
            h0 = h0 * u0
        else:
            h0 = h0 * self.dropout_rate

        z2 = np.matmul(h0, self.w2)
        y_hat = F.softmax(z2)

        parameters = {
            "z1": z1,
            "h0": h0,
            "z2": z2,
            "y_hat": y_hat
        }

        if train_mode:
            parameters['u0'] = u0
          #  parameters['ux'] = ux

        return parameters

    def backward(self, x, y, parameters):
        # Back propagation
      #  ux = parameters['ux']
        z1 = parameters["z1"]
        h0 = parameters["h0"]
        u0 = parameters["u0"]
        z2 = parameters["z2"]
        y_hat = parameters["y_hat"]

        # Gradients Calculation
        grad_l_by_z2 = y_hat - y
        grad_z2_by_w2 = h0

        grad_l_by_w2 = np.matmul(np.reshape(grad_l_by_z2, (self.out_len, 1)),
                                 np.reshape(grad_z2_by_w2, (1, self.h0_len))).T

        grad_z2_by_h0 = self.w2
        grad_h0_by_z1 = F.differentiate_relu(z1) * u0
        grad_z1_by_w1 = x #* ux

        grad_l_by_w1 = np.matmul(np.reshape(grad_l_by_z2, (1, self.out_len)),
                                 grad_z2_by_h0.T)
        grad_l_by_w1 = grad_l_by_w1 * grad_h0_by_z1
        grad_l_by_w1 = np.matmul(grad_l_by_w1.reshape((self.h0_len, 1)),
                                 grad_z1_by_w1.reshape((1, self.input_len))).T

        grads = {"w1": grad_l_by_w1,
                 "w2": grad_l_by_w2}

        return grads

    def train(self, train_x, train_y, batch_size=20, num_of_epochs=100, lr_decay=0.0):
        for epoch in range(num_of_epochs):
            loss_sum = 0.0
            correct_predictions = 0

            for i in range(batch_size, len(train_x), batch_size):
                # Get the batch inputs and outputs
                batch_x = train_x[i - batch_size:min(i, len(train_x)), :]
                batch_y = train_y[i - batch_size:min(i, len(train_y)), :]

                # Variables to sum the gradients
                grad_w1_sum = np.zeros((self.input_len, self.h0_len))
                grad_w2_sum = np.zeros((self.h0_len, self.out_len))

                # Iterate through batch
                for j in range(len(batch_x)):
                    x = batch_x[j]
                    y = batch_y[j]

                    param = self.forward(x,train_mode=True)
                    grads = self.backward(x, y, param)

                    # Calculate loss
                    loss = F.log_loss(y, param['y_hat'])

                    # Check if predicted correctly
                    y_hat = param["y_hat"]
                    y_pred = np.argmax(y_hat)
                    y_true = np.argmax(y)
                    correct_predictions += (y_pred == y_true)

                    # Add loss and gradients to sum
                    loss_sum += loss
                    grad_w1_sum = np.add(grad_w1_sum, grads["w1"])
                    grad_w2_sum = np.add(grad_w2_sum, grads["w2"])

                # Average the gradients
                grad_w1_avg = grad_w1_sum / len(batch_x)
                grad_w2_avg = grad_w2_sum / len(batch_x)

                # Update the weights
                self.w1 = self.w1 - grad_w1_avg * self.learning_rate
                self.w2 = self.w2 - grad_w2_avg * self.learning_rate

            # Learning rate decay
            self.learning_rate -= lr_decay

            # Print info
            print("Epoch number: %d" % (epoch + 1))
            print("Loss: %f" % (loss_sum / len(train_x)))
            print("Accuracy: %f" % (correct_predictions / len(train_x)))

    def test(self, test_x, test_y):
        loss_sum = 0.0
        correct_predictions = 0
        for i, x in enumerate(test_x):
            y = test_y[i]

            param = self.forward(x, train_mode=False)
            # Calculate loss
            y_hat = param['y_hat']
            loss_sum += F.log_loss(y, y_hat)

            y_pred = np.argmax(y_hat)
            y_true = np.argmax(y)
            correct_predictions += (y_pred == y_true)

        print("Test Results")
        print("Loss: %f" % (loss_sum / len(test_x)))
        print("Accuracy: %f" % (correct_predictions / len(test_x)))
		    def relu(h):
        return np.maximum(0, h)
    def softmax(z):
        # Calculate exponent term first
        ex = np.exp(z)
        return ex / np.sum(ex, axis=0, keepdims=True)
    def log_loss(y, y_hat):
        return np.sum(-y * np.log(y_hat.clip(min=1e-6)))
    def differentiate_relu(x):
        return (x > 0) * 1
    def dropout(h, p):
        return np.random.binomial(1, p, size=len(h)) / p