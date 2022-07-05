from cvxpy import *
import numpy as np

def problem2_d():
    #values setting
    n = 3
    eps = 0.1

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    b = np.array([1, 1, 1])

    #Variable
    x = Variable(n)

    #constraints
    constraints = [norm((A@x - b), 1) <= eps]

    #objective function
    obj = Minimize(length(x))

    #solve the problems
    prob = Problem(obj, constraints)
    prob.solve(qcp = True)

    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", x.value) 

def problem2_e():
    #values setting
    n = 4
    eps = 0.1

    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 16, 15],
                  [17, 18, 20, 19]])

    b = np.array([1, 2, 4, 3, 5])

    #Variable
    x = Variable(n)

    #constraints
    constraints = [norm((A@x - b), 1) <= eps]

    #objective function
    obj = Minimize(length(x))

    #solve the problems
    prob = Problem(obj, constraints)
    prob.solve(qcp = True)

    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", x.value) 

if __name__ == "__main__":
    problem2_d()
    problem2_e()
