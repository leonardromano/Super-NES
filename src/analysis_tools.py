#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:56:45 2022

@author: leonard
"""

from numpy import pi

import src.Euler_solver_1D as main

def total_energy(x, P, G, U, GAMMA, Nalpha, Nu):
    for i in range(Nu):
        for j in range(Nalpha):
            Pij = P[i, j]
            Gij = G[i, j]
            Uij = U[i, j]
            
            Ek  = 0.5 * Gij * Uij**2
            Eth = Pij / (GAMMA - 1)
            E   = main.trapz(x, x**2 * (Ek + Eth))
            
            Cx = (16 * pi * E / 25)**(-0.2)
            print("Cx(%d, %d) = %g"%(i, j, Cx))
            
            
def total_momentum(x, G, U, Nalpha, Nu):
    for i in range(Nu):
        for j in range(Nalpha):
            Gij = G[i, j]
            Uij = U[i, j]
    
            I   = 4 * main.trapz(x, x**2 * Gij * Uij)
            print("Ip(%d, %d) = %g"%(i, j, I))

def thickness(x, G, Nalpha, Nu, Nx):
    for i in range(Nu):
        for j in range(Nalpha):
            Gij = G[i, j]
            
            fG = Gij / main.trapz(x, x**2 * Gij)
            
            frac = 0.0
            k = 0
            while frac < 0.99:
                k   +=1
                i0   = Nx - 1 - k
                xij  = x[i0:]  
                frac =  main.trapz(xij, xij**2 * fG[i0:])
            
            xi = x[Nx - 1 - k]
            print("d(%d, %d) = %g"%(i, j, 1 - xi))
