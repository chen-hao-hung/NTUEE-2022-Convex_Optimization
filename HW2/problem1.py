#Convex Optimization HW2
from cvxpy import *
import numpy as np

def problem1():
    #variables
    x = Variable(3)
    P0 = np.array([[8.2, -3.8, 8.6],
                   [-3.8, 1.8, -4.3],
                   [8.6, -4.3, 11.9]])

    P1 = np.array([[1.2, 0.3, -0.6],
                   [0.3, 1.7, 2.4],
                   [-0.6, 2.4, 5.0]])
    
    P2 = np.array([[1.5, 1.3, -1.9],
                   [1.3, 10.1, -3.2],
                   [-1.9, -3.2, 3.8]])

    A = np.array([[1, 2, -3],
                  [-7, 8, -9]])
    
    b = np.array([1.5, 8])

    q0 = np.transpose(np.array([7, 8, 9]))

    q1 = np.transpose(np.array([4, 5, 6]))
    
    q2 = np.transpose(np.array([1, 2, 3]))

    r0, r1, r2 = 1, 2, 3

    #constraints
    constraints = [A@x == b, (1/2)*quad_form(x, P1)+q1@x+r1 <= 0, (1/2)*quad_form(x, P2)+q2@x+r2 <= 0]

    #objective function
    obj = Minimize((1/2)*quad_form(x, P0) + q0@x + r0)

    #solve the problems
    prob = Problem(obj, constraints)
    prob.solve()
    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", x.value)
    print("dual variable", constraints[0].dual_value, constraints[1].dual_value, constraints[2].dual_value)
    
if __name__ == '__main__':
    problem1()

