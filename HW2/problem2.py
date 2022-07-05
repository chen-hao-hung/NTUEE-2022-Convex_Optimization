#Convex Optimization HW2
from itertools import product
import numpy as np
from cvxpy import *

def problem2_d():
    xWx = []
    x = np.array(list(product((1, -1), repeat = 5)))
    W1 = np.array([[-0.8, 0.2, 0.3, 1.0, -0.3],
                   [0.2, -0.1, -0.8, 1.3, 1.1],
                   [0.3, -0.8, -1.0, -0.2, 0.4],
                   [1.0, 1.3, -0.2, 0.9, 0.8],
                   [-0.3, 1.1, 0.4, 0.8, 0.9]])

    for i in range(len(x)):
        xWx.append(np.transpose(x[i]).dot(W1).dot(x[i]))
        tmp = np.min(xWx)
        index = xWx.index(tmp)

    print (tmp, x[index])
    

print('problem2_d :')
problem2_d()

def problem2_e():
    n = 5
    v = Variable(n)

    W1 = np.array([[-0.8, 0.2, 0.3, 1.0, -0.3],
                   [0.2, -0.1, -0.8, 1.3, 1.1],
                   [0.3, -0.8, -1.0, -0.2, 0.4],
                   [1.0, 1.3, -0.2, 0.9, 0.8],
                   [-0.3, 1.1, 0.4, 0.8, 0.9]])

    constraints = [W1+diag(v) >> 0]
    obj = Maximize((-1)*(np.ones(n))@v)
    prob = Problem(obj, constraints)
    prob.solve()
    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", v.value)

print('problem2_e :')
problem2_e()

def problem2_g_d():
    xWx = []
    x = np.array(list(product((1, -1), repeat = 10)))
    W2 = np.array([[0.7, -0.3, 0.3, 0.3, 0.1, 0.6, -0.6, -1.3, 1.0, 0.1],
                   [-0.3, 0.4, 0.4, 0, -0.6, -0.1, 0, 1.2, -0.3, 0.4],
                   [0.3, 0.4, 0.1, 0, -0.6, -0.3, 0.5, 0.2, 1.2, 0.3],
                   [0.3, 0, 0, 0.2, 0.3, 0.1, 0.6, 1.9, 0.5, 0.5],
                   [0.1, -0.6, -0.6, 0.3, -1.6, -1.2, -0.2, 0.3, 0.7, -0.9],
                   [0.6, -0.1, -0.3, 0.1, -1.2, 0.9, 0.7, -0.8, -0.8, -0.4],
                   [-0.6, 0, 0.5, 0.6, -0.2, 0.7, -0.9, -0.5, 0, -0.3],
                   [-1.3, 1.2, 0.2, 1.9, 0.3, -0.8, -0.5, -1.7, -0.1, 0.3],
                   [1.0, -0.3, 1.2, 0.5, 0.7, -0.8, 0, -0.1, 1.9, 1.2],
                   [0.1, 0.4, 0.3, 0.5, -0.9, -0.4, -0.3, 0.3, 1.2, -0.3]])

    for i in range(len(x)):
        xWx.append(np.transpose(x[i]).dot(W2).dot(x[i]))
        tmp = np.min(xWx)
        index = xWx.index(tmp)

    print (tmp, x[index])
    

print('problem2_g_d :')
problem2_g_d()

def problem2_g_e():
    n = 10
    v = Variable(n)
    W2 = np.array([[0.7, -0.3, 0.3, 0.3, 0.1, 0.6, -0.6, -1.3, 1.0, 0.1],
                   [-0.3, 0.4, 0.4, 0, -0.6, -0.1, 0, 1.2, -0.3, 0.4],
                   [0.3, 0.4, 0.1, 0, -0.6, -0.3, 0.5, 0.2, 1.2, 0.3],
                   [0.3, 0, 0, 0.2, 0.3, 0.1, 0.6, 1.9, 0.5, 0.5],
                   [0.1, -0.6, -0.6, 0.3, -1.6, -1.2, -0.2, 0.3, 0.7, -0.9],
                   [0.6, -0.1, -0.3, 0.1, -1.2, 0.9, 0.7, -0.8, -0.8, -0.4],
                   [-0.6, 0, 0.5, 0.6, -0.2, 0.7, -0.9, -0.5, 0, -0.3],
                   [-1.3, 1.2, 0.2, 1.9, 0.3, -0.8, -0.5, -1.7, -0.1, 0.3],
                   [1.0, -0.3, 1.2, 0.5, 0.7, -0.8, 0, -0.1, 1.9, 1.2],
                   [0.1, 0.4, 0.3, 0.5, -0.9, -0.4, -0.3, 0.3, 1.2, -0.3]])
    ones = np.ones((10, 1))
    constraints = [W2+diag(v) >> 0]
    obj = Maximize(-(ones.T)@v)
    prob = Problem(obj, constraints)
    prob.solve()
    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", v.value)

print('problem2_g_e :')
problem2_g_e()

