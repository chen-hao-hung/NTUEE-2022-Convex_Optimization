#Convex Optimization HW3
#problem3_e
from cvxpy import *
import numpy as np
def problem3_e():
    n = 3
    x = Variable(n)
    F1 = np.array([[0.9, -0.4, 0.6],
                   [-0.4, 3.9, -1.0],
                   [0.6, -1.0, 2.7]])
    F2 = np.array([[0.5, 0.8, -1.0],
                   [0.8, 1.7, -2.6],
                   [-1.0, -2.6, 5.4]])
    F3 = np.array([[0.2, 0.1, 0.6],
                   [0.1, 0.6, -0.1],
                   [0.6, -0.1, 3.3]])
    G = np.array([[-1.2, 0.5, -1.2],
                   [0.5, -1.4, -1.0],
                   [-1.2, -1.0, -3.3]])
    c = np.transpose(np.array([-4, -2, -4]))

    constraints = [G+x[0]*F1+x[1]*F2+x[2]*F3 << 0]
    obj = Minimize(c@x)
    prob = Problem(obj, constraints)
    prob.solve()

    print ("status:", prob.status)
    print ("optimal value", prob.value)
    print ("optimal var", x.value)

print('problem3_e :')
problem3_e()

#problem3_i
x = np.array([0, 0, 0])
t = 1
F1 = np.array([[0.9, -0.4, 0.6],
               [-0.4, 3.9, -1.0],
               [0.6, -1.0, 2.7]])
F2 = np.array([[0.5, 0.8, -1.0],
               [0.8, 1.7, -2.6],
               [-1.0, -2.6, 5.4]])
F3 = np.array([[0.2, 0.1, 0.6],
               [0.1, 0.6, -0.1],
               [0.6, -0.1, 3.3]])
F = np.array([F1, F2, F3])
G = np.array([[-1.2, 0.5, -1.2],
               [0.5, -1.4, -1.0],
               [-1.2, -1.0, -3.3]])
c = np.transpose(np.array([-4, -2, -4]))

def my_objective(x, t, c, F, G):
    f0 = np.matmul(np.transpose(c), x)
    f1 = G+x[0]*F[0]+x[1]*F[1]+x[2]*F[2]
    f1_inv = np.linalg.inv(f1)
    obj = t*f0-np.log(np.linalg.det(-f1))
    gradient = t*c+np.transpose(np.array([np.trace(-np.matmul(f1_inv, F[0])), np.trace(-np.matmul(f1_inv, F[1])), np.trace(-np.matmul(f1_inv, F[2]))]))
    Hessian = np.array([[np.trace(np.matmul(np.matmul(f1_inv, F[0]), np.matmul(f1_inv, F[0]))), np.trace(np.matmul(np.matmul(f1_inv, F[0]), np.matmul(f1_inv, F[1]))), np.trace(np.matmul(np.matmul(f1_inv, F[0]), np.matmul(f1_inv, F[2])))],
                        [np.trace(np.matmul(np.matmul(f1_inv, F[1]), np.matmul(f1_inv, F[0]))), np.trace(np.matmul(np.matmul(f1_inv, F[1]), np.matmul(f1_inv, F[1]))), np.trace(np.matmul(np.matmul(f1_inv, F[1]), np.matmul(f1_inv, F[2])))],
                        [np.trace(np.matmul(np.matmul(f1_inv, F[2]), np.matmul(f1_inv, F[0]))), np.trace(np.matmul(np.matmul(f1_inv, F[2]), np.matmul(f1_inv, F[1]))), np.trace(np.matmul(np.matmul(f1_inv, F[2]), np.matmul(f1_inv, F[2])))]])
    return obj, gradient, Hessian

#problem3_j
def centrality_problem(x_init, t, c, F, G, alpha=0.1, beta=0.7):
    x = x_init
    s = 1
    repeat_inner = True
    eps_inner = 1e-8
    while(repeat_inner):
        gradient, Hessian = my_objective(x, t, c, F, G)[1], my_objective(x, t, c, F, G)[2]
        newton_step = -1*np.linalg.solve(Hessian, gradient)
        newton_decrement = (-1*np.matmul(gradient, newton_step))**0.5

        while(my_objective(x+s*newton_step, t, c, F, G)[0] > (my_objective(x, t, c, F, G)[0]-alpha*s*newton_decrement**2)):
            s = beta*s

        #print(f"x: {x}")
        #print(f"objective: {my_objective(x, t, c, F, G)[0]}")
        #print(f"decrement: {newton_decrement}\n")
        x = x+s*newton_step
        if newton_decrement**2/2 < eps_inner:
            repeat_inner = False
    return x, newton_decrement**2/2

#problem3_(k)-(o)
def interior_method(mu=2):
    eps_outer = 1e-8
    eps_inner = 1e-8
    repeat_outer = True
    outer_iteration = 0
    t = 1
    mu = mu
    k = 0
    x = np.array([0,0,0])
    x_star_list = []
    decrement_2_2_list = []
    k_list = []
    l_list = []
    while(repeat_outer):
        repeat_inner = True
        inner_iteration = 0
        print(f"outer iteration: {outer_iteration+1}")
        print(f"t: {t}")

        while(repeat_inner):
            alpha=0.1
            beta=0.7
            s = 1
            gradient, Hessian = my_objective(x, t, c, F, G)[1], my_objective(x, t, c, F, G)[2]
            newton_step = -1*np.linalg.solve(Hessian, gradient)
            newton_decrement = (-1*np.matmul(gradient, newton_step))**0.5
            
            while(np.linalg.det(-(G+(x+s*newton_step)[0]*F[0]+(x+s*newton_step)[1]*F[1]+(x+s*newton_step)[2]*F[2]))) <= 0:
                s = beta*s
            while(my_objective(x+s*newton_step, t, c, F, G)[0] > (my_objective(x, t, c, F, G)[0]-alpha*s*newton_decrement**2)):
                s = beta*s
            
            #objective_list.append(my_objective(x, t, c, F, G)[0])
            decrement_2_2_list.append((newton_decrement**2)/2)
            k_list.append(k)

            x = x+s*newton_step
            inner_iteration += 1
            k += 1
            if newton_decrement**2/2 <= eps_inner:
                repeat_inner = False
        
        print(f"inner iteration: {inner_iteration+1}")
        print(f"objective function value: {my_objective(x, t, c, F, G)[0]}")
        print("====================================================================\n")
        outer_iteration += 1
        l_list.append(outer_iteration)
        x_star_list.append(x)
        t = mu*t
        if 3/t < eps_outer:
            repeat_outer = False
            
    print(f"total iteration: {k}")
    print(f"Optimal point: {x}")
    
interior_method(mu=2)