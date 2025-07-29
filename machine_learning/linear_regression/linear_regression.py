def loss_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight * radio[i] + bias)) ** 2
    return total_error / companies


def updata_weights(radio, sales, weight, bias, lr):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        weight_deriv += -2 * radio[i] * (sales[i] - (weight * radio[i] + bias))
        bias_deriv += -2 * (sales[i] - (weight * radio[i] + bias))

    # subtract because the derivatives point in direction of steepest ascent
    # param -= learning_rate * param.grad
    weight -= lr * (weight_deriv / companies)
    bias -= lr * (bias_deriv / companies)

    return weight, bias


def training(radio, sales, weight, bias, lr, iters):
    loss_history = []

    for i in range(iters):
        weight, bias = updata_weights(radio, sales, weight, bias, lr)

        loss = loss_function(radio, sales, weight, bias)
        loss_history.append(loss)

        # Log Progress
        if i % 10 == 0:
            print(
                "iter={:d}    weight={:.2f}    bias={:.4f}    loss={:.2}".format(
                    i, weight, bias, loss
                )
            )

    return weight, bias, loss_function
