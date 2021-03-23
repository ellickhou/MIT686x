import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
    ## 0 = 1/n(-b + A dot theta_head) + lambda_factor*theta_head
    ## 1/n*b = 1/n A dot theta_head + I*lamda dot theta_head
    ## theta_hat = matmul((A+lamda*I)^-1,b) = (d+1,d+1) (d+1, )= (d+1,)
    ## b = 1/n*sum(X[i]*y[i]) = matmul(X.T, Y) = (d+1,n) (n, ) = (d+1,)
    ## A = 1/n*matmul(X.T, X) = (d+1,d+1)
    b = np.matmul(X.T, Y)
    A = np.matmul(X.T, X)
    L_array = np.identity(X.shape[1])*lambda_factor
    A_L = A + L_array
    theta_hat = np.matmul(np.linalg.pinv(A_L), b)
    # theta_hat = np.matmul(np.linalg.inv(A_L), b)
    return theta_hat
    # raise NotImplementedError

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
