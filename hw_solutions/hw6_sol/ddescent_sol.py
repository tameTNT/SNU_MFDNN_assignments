import matplotlib.pyplot as plt 
import numpy as np 
from math import sqrt 
import time


"""
Step 1 : Generate Toy data
"""

d = 35
n_train, n_val, n_test = 300, 60, 30
np.random.seed(0)
beta = np.random.randn(d)
beta_true = beta / np.linalg.norm(beta)
# Generate and fix training data
X_train = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_train)])
Y_train = X_train @ beta_true + np.random.normal(loc = 0.0, scale = 0.5, size = n_train)
# Generate and fix validation data (for tuning lambda). 
X_val = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_val)])
Y_val = X_val @ beta_true 
# Generate and fix test data
X_test = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_test)])
Y_test = X_test @ beta_true 

# Generate list of lambda values from which we search for the optimal lambda. 


"""
Step 2 : Implement basic functionalities.
"""

def solve_lsqr(X_tilde, Y, lamda):
    """
    Return optimal theta value that corresponds to given X_tilde, Y, and lambda. 
    """
    # return  np.linalg.inv(np.dot(X.T, X) + lamda * np.identity(X.shape[1])) @ X.T @ Y    
    # using backsolve is faster than computing explicit inverse
    return np.linalg.solve(
        np.dot(X_tilde.T, X_tilde) + lamda * np.identity(X_tilde.shape[1]), X_tilde.T @ Y
    )


def get_error(tilde_X, Y, theta) : 
    """
    Calculate $RMSE$ for given test dataset (tilde_x, y) and infered theta vector. 
    {(tilde_x, y)} may be either the validation data or the test data.
    """
    return np.linalg.norm(tilde_X @ theta - Y) 


def optimize_lambda(X_tilde, Y, X_tilde_val, Y_val) :
    """
    Optimize lambda value for a given tilde_x and y. 
    @param X_tilde: ReLU feature of training data. 
    @param X_tilde_val: ReLU feature of validation data. 
    """
    lambda_list = [2 ** i for i in range(-6, 6)]

    val_errors = []
    for lamda in lambda_list :
        theta = solve_lsqr(X_tilde, Y, lamda)
        val_error = get_error(X_tilde_val, Y_val, theta)
        val_errors.append(val_error)
    optimal_lambda = lambda_list[np.argmin(val_errors)]  
    return optimal_lambda 


def ReLU(x) : 
    """
    Custom ReLU function that is applied elementwisely.
    """
    return np.where(np.asarray(x) > 0, x, 0)


errors_opt_lambda = []
errors_fixed_lambda = []


# Different number of parameters 
num_params = np.arange(1,1501,10)

start = time.time()

for p in num_params : 
    weight_matrix = np.random.normal(loc = 0.0, scale = 1 / sqrt(p), size = (p, d))
    
    # ReLU feature of training and validation data 
    X_tilde = ReLU(X_train @ weight_matrix.T) 
    X_tilde_val = ReLU(X_val @ weight_matrix.T)
    
    # optimize lambda 
    optimal_lambda = optimize_lambda(X_tilde, Y_train, X_tilde_val, Y_val)
    
    # theta value inferred from optimal lambda value
    theta_opt_lambda = solve_lsqr(X_tilde, Y_train, optimal_lambda)   
    
    # theta value inferred from fixed lambda value
    theta_fixed_lambda = solve_lsqr(X_tilde, Y_train, 0.01) 
    
    # ReLU feature of test data 
    X_tilde_test = ReLU(X_test @ weight_matrix.T)
    
    # Test errors 
    error_opt_lambda = get_error(X_tilde_test, Y_test, theta_opt_lambda)
    error_fixed_lambda = get_error(X_tilde_test, Y_test, theta_fixed_lambda)
    errors_opt_lambda.append(error_opt_lambda)
    errors_fixed_lambda.append(error_fixed_lambda)
    
    
end = time.time()
print("Time ellapsed in training is: {}".format(end - start))
    


"""
Step 3 : Plot the results
"""    

plt.figure(figsize = (24, 8))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('font', size = 24)


plt.scatter(num_params, errors_fixed_lambda, color = 'black',
            label = r"Test error with fixed $\lambda = 0.01$",
            ) 
plt.legend()

plt.plot(num_params, errors_opt_lambda, 'k', label = r"Test error with tuned $\lambda$")
plt.legend()
plt.xlabel(r'$\#$ parameters')
plt.ylabel('Test error')
plt.title(r'Test error vs. $\#$ params')

plt.savefig('double_descent.png')
plt.show()