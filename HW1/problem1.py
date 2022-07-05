from cvxpy import *
import numpy as np

def problem1():
    #variables
    x = Variable(nonneg = True)
    y = Variable(nonneg = True()

    #constraints
    constraints = [x + y >= 1, inv_pos(x) + inv_pos(y) <= 1, power(x, 1.5) + power(y, 1.5) <= power(5, 1.5)]

    #objective function
    obj = Minimize(quad_over_lin(x, y))

    #solve the problems
    prob = Problem(obj, constraints)
    prob.solve()
    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", x.value, y.value)

if __name__ == '__main__':
    problem1()
