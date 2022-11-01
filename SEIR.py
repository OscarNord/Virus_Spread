# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:34:30 2020

@author: Oscar
"""
import numpy as np
import matplotlib.pyplot as plt
from ODESolver import RungeKutta4


## a)

class Region:
    
    def __init__(self,name,S_0,E2_0):
        self.name = name
        self.S_0 = S_0
        self.E1_0 = 0
        self.E2_0 = E2_0
        self.I_0 = 0
        self.Ia_0 = 0
        self.R_0 = 0
        self.population = self.S_0+self.E1_0+self.E2_0+self.I_0+self.Ia_0+self.R_0
    
    def set_SEIR_values(self,u,t):
        self.t = t
        self.S = u[:,0]; self.E1 = u[:,1]; self.E2 = u[:,2]
        self.I = u[:,3]; self.Ia = u[:,4]; self.R = u[:,5]
        
    def plot(self):
        plt.xlabel('Time(days)')
        plt.ylabel('Population')
        plt.title(f'{self.name}')
        plt.plot(self.t, self.S,     label='S')
        plt.plot(self.t, self.I, label='I')
        plt.plot(self.t, self.Ia, label='Ia')
        plt.plot(self.t, self.R, label='R')


## b)

class ProblemSEIR:
    
    def __init__(self, region, beta, r_ia = 0.1, r_e2=1.25,
        lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        if isinstance(beta, (float, int)): # number?
            self.beta = lambda t: beta # wrap as function
        elif callable(beta):
            self.beta = beta
    
        self.region = region
        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu
        
        
    def set_initial_condition(self):
        self.initial_condition = \
            [self.region.S_0,self.region.E1_0,\
             self.region.E2_0,self.region.I_0,\
             self.region.Ia_0,self.region.R_0]
    
    def get_population(self):
        self.population = self.region.population
        return self.population
        
    def solution(self,u,t):
        self.region.set_SEIR_values(u,t)
       
    def __call__(self,u,t):
        S, E1, E2, I, Ia, R = u
        N = self.population
        
        r_ia = self.r_ia; r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1; lmbda_2 = self.lmbda_2
        p_a = self.p_a; mu = self.mu
        beta = self.beta
        
        dS  = -beta(t)*S*I/N - r_ia*beta(t)*S*Ia/N - r_e2*beta(t)*S*E2/N
        dE1 = beta(t)*S*I/N + r_ia*beta(t)*S*Ia/N + r_e2*beta(t)*S*E2/N - lmbda_1*E1
        dE2 = lmbda_1*(1-p_a)*E1 - lmbda_2*E2
        dI  = lmbda_2*E2 - mu*I
        dIa = lmbda_1*p_a*E1 - mu*Ia
        dR  = mu*(I + Ia)
        return [dS, dE1, dE2, dI, dIa, dR]


class SolverSEIR:
    
    def __init__(self,problem,T,dt):
        self.problem = problem
        self.T = T
        self.dt = dt
        self.total_population = problem.get_population()
            
    def solve(self, method=RungeKutta4):
        solver = method(self.problem)
        solver.set_initial_condition(self.problem.initial_condition)
        n = int(self.T/self.dt)
        t = np.linspace(0,self.T,n+1)
        u, t = solver.solve(t)
        # Send the values of S, E1, E2, I, Ia, R, and t
        # from the Problem class to the Region class:
        self.problem.solution(u, t)
        
    
if __name__ == '__main__':

    #1.2 a), class Region:
    nor = Region('Norway',S_0=5e6,E2_0=100)
    print(nor.name, nor.population)
    S_0, E1_0, E2_0 = nor.S_0, nor.E1_0, nor.E2_0
    I_0, Ia_0, R_0 = nor.I_0, nor.Ia_0, nor.R_0
    print(f'S_0 = {S_0}, E1_0 = {E1_0}, E2_0 = {E2_0}')
    print(f'I_0 = {I_0}, Ia_0 = {Ia_0}, R_0 = {R_0}')
    u = np.zeros((2,6))
    u[0,:] = [S_0, E1_0, E2_0, I_0, Ia_0, R_0]
    nor.set_SEIR_values(u,0)
    print(nor.S, nor.E1, nor.E2, nor.I, nor.Ia, nor.R)

    #1.2 b) class ProblemSEIR:
    problem = ProblemSEIR(nor,beta=0.5)
    problem.set_initial_condition()
    print(problem.initial_condition)
    print(problem.get_population())
    print(problem([1,1,1,1,1,1],0))

    #1.3 c) class SolverSEIR:
    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve()
    nor.plot()
    plt.legend()
    plt.show()
            
"""
Terminal>python SEIR.py
Norway 5000100.0
S_0 = 5000000.0, E1_0 = 0, E2_0 = 100
I_0 = 0, Ia_0 = 0, R_0 = 0
[5000000.       0.] [0. 0.] [100.   0.] [0. 0.] [0. 0.] [0. 0.]
[5000000.0, 0, 100, 0, 0, 0]
5000100.0
[-2.3499530009399813e-07, -0.3299997650046999, -0.302, 0.3, -0.068, 0.4]
"""