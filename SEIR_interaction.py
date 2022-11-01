# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:54:18 2020

@author: Oscar
"""

import numpy as np
import matplotlib.pyplot as plt
from SEIR import ProblemSEIR, Region, SolverSEIR

## a)

class RegionInteraction(Region):
    
        def __init__(self,name,S_0,E2_0,lat,long):
            super().__init__(name,S_0,E2_0)
            self.lat = lat*(np.pi/180)
            self.long = long*(np.pi/180)
            
        
        def distance(self,other):
            iFi= self.lat; jFi = other.lat
            iLmbda = self.long; jLmbda = other.long
            RE = 64                                 # Given in units of 10^5m
            arg = np.sin(iFi)*np.sin(jFi)+np.cos(iFi)*np.cos(jFi)*np.cos(abs(iLmbda-jLmbda))
            if arg > 1 or arg < 0:
                raise ValueError('Argument for arccos was not defined in the interval (0,1)')
                exit()
            dSigma = np.arccos(arg)
            return RE* dSigma 
    
## b)

class ProblemInteraction(ProblemSEIR):

    def __init__(self, region, area_name, beta, r_ia = 0.1, r_e2=1.25,
    lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        super().__init__(region,beta,r_ia,r_e2,lmbda_1,lmbda_2,p_a,mu)
        self.area_name = area_name
        
    def get_population(self):
        totPop = 0
        for each in self.region:
            totPop += each.population
        return totPop

    def set_initial_condition(self):
        region = self.region
        self.initial_condition = []
        for each in region:
            ic = [each.S_0,each.E1_0,each.E2_0,each.I_0,each.Ia_0,each.R_0]
            self.initial_condition += ic
        
    
    
    def __call__(self, u, t):
        n = len(self.region)
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = []
        r_ia = self.r_ia; r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1; lmbda_2 = self.lmbda_2
        p_a = self.p_a; mu = self.mu
        beta = self.beta;
        
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            Ni = sum(SEIR_list[i])
            dS = 0                
            sumE2 = 0 
            sumIa = 0
            
            for j in range(n):
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                Nj = sum(SEIR_list[j])
                dij = self.region[i].distance(self.region[j])
                sumE2 += E2_other*np.exp(-dij)/Nj
                sumIa += Ia_other*np.exp(-dij)/Nj
                
            dS = -beta(t)*S*I/Ni - r_ia*beta(t)*S*sumIa - r_e2*beta(t)*S*sumE2
            dE1 = -dS - lmbda_1*E1
            dE2 = lmbda_1*(1-p_a)*E1 - lmbda_2*E2
            dI  = lmbda_2*E2 - mu*I
            dIa = lmbda_1*p_a*E1 - mu*Ia
            dR  = mu*(I + Ia)
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative
    
    def solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n)
        self.I = np.zeros(n)
        self.Ia = np.zeros(n)
        self.R = np.zeros(n)
        
        SEIR_list = [u[:, i:i+6] for i in range(0, n_reg*6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += SEIR[:,0]
            self.E1 += SEIR[:,1]
            self.E2 += SEIR[:,2]
            self.I += SEIR[:,3]
            self.Ia += SEIR[:,4]
            self.R += SEIR[:,5]

    def plot(self):
        plt.xlabel('Time(days)')
        plt.ylabel('Population')
        plt.title(f'{self.area_name}')
        plt.plot(self.t, self.S, label='S')
        plt.plot(self.t, self.I, label='I')
        plt.plot(self.t, self.Ia, label='Ia')
        plt.plot(self.t, self.R, label='R')

    
if __name__ == '__main__':
    innlandet = RegionInteraction('Innlandet',S_0=371385, E2_0=0, \
    lat=60.7945,long=11.0680)
    oslo = RegionInteraction('Oslo',S_0=693494,E2_0=100, \
    lat=59.9,long=10.8)
    print(oslo.distance(innlandet))

    problem = ProblemInteraction([oslo,innlandet],'Norway_east', beta=0.5)
    print(problem.get_population())
    problem.set_initial_condition()
    print(problem.initial_condition) #non-nested list of length 12
    u = problem.initial_condition
    print(problem(u,0)) #list of length 12. Check that values make sense
    
    
    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve()
    problem.plot()
    plt.legend()
    plt.show()

"""
Terminal>python SEIR_interaction.py
1.0100809386280782
1064979
[693494, 0, 100, 0, 0, 0, 371385, 0, 0, 0, 0, 0]
[-62.49098896472576, 62.49098896472576, -50.0, 50.0, 0.0, 0.0, -12.187832324277787, 12.187832324277787, 0.0, 0.0, 0.0, 0.0]
"""
        
 