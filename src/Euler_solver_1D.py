#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:28:42 2022

@author: leonard
"""

from numpy import zeros, linspace, where, absolute, exp

from numba import njit

@njit
def trapz(x, y):
    Nx = x.shape[0]
    s = 0
    for i in range(1, Nx):
        s += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return s

@njit
def grad(u, dx):
    du_dx     = zeros(u.shape[0])
    du_dx[0]  = u[1] - u[0]
    du_dx[-1] = u[-1] - u[-2]
    for i in range(1, u.shape[0]-1):
        du_dx[i] = 0.5 * (u[i+1] - u[i-1])
    du_dx /= dx
    return du_dx

@njit
def gradients(x, V, G, Z, GAMMA, beta, alpha):
    y   = Z / ((1 - V)**2 - Z)
    VH  = 1 / beta / GAMMA + V * (1 - V) * (1 / alpha - V) / 3 / Z
    dV_dlogx = 3 * y * (V - VH)
    
    dlogG_dlogx = (dV_dlogx + 3 * V) / (1 - V)
    dlogZ_dlogx = - (3 / beta + 2 * (1 - V) + (GAMMA - 1) * dV_dlogx) / (1 - V)
    return dV_dlogx, dlogG_dlogx, dlogZ_dlogx


def compile_funcs():
    #array of floating point numbers
    arr = linspace(0.5, 0.75, 3)
    #add all functions that need to be compiled here
    gradients(1., 0.5, 1., 1., 1., 1., 1.)
    grad(arr, 1.0)
    trapz(arr, arr)

@njit
def RK4(x, dlogx, V, G, Z, GAMMA, beta, alpha):
    dV1, dlogG1, dlogZ1 = gradients(x, V, G, Z, GAMMA, beta, alpha)

    #get dx
    C_CFL   = 1e-3  #CFL parameter
    if absolute(V) > C_CFL:
        dlogx_V = C_CFL * absolute(V / (dV1 + 1e-37))
    else:
        dlogx_V = C_CFL * absolute(1 / (dV1 + 1e-37) / beta / GAMMA)
    dlogx_G = C_CFL * absolute(1 / dlogG1)
    dlogx_Z = C_CFL * absolute(1 / dlogZ1)

    dlogx = min(dlogx, dlogx_V, dlogx_G, dlogx_Z)
    
    #multiply through with 0.5 * dlogx
    dV1    *= dlogx
    dlogG1 *= dlogx
    dlogZ1 *= dlogx
    
    #RK4 compute other gradients
    dV2, dlogG2, dlogZ2 = gradients(x * (1 - 0.5 * dlogx), \
                                    V - 0.5 * dV1, \
                                    G * (1 - 0.5 * dlogG1), \
                                    Z * (1 - 0.5 * dlogZ1), \
                                    GAMMA, beta, alpha)
    dV2    *= dlogx
    dlogG2 *= dlogx
    dlogZ2 *= dlogx
        
    dV3, dlogG3, dlogZ3 = gradients(x * (1 - 0.5 * dlogx), \
                                    V - 0.5 * dV2, \
                                    G * (1 - 0.5 * dlogG2), \
                                    Z * (1 - 0.5 * dlogZ2), \
                                    GAMMA, beta, alpha)
    dV3    *= dlogx
    dlogG3 *= dlogx
    dlogZ3 *= dlogx
    
    dV4, dlogG4, dlogZ4 = gradients(x * (1 - dlogx), V - dV3, \
                                    G * (1 - dlogG3), Z * (1 - dlogZ3), \
                                    GAMMA, beta, alpha)
    
    dV4    *= dlogx
    dlogG4 *= dlogx
    dlogZ4 *= dlogx
        
    dV    = 1/6 * (dV1    + 2 * dV2    + 2 * dV3    + dV4)
    dlogG = 1/6 * (dlogG1 + 2 * dlogG2 + 2 * dlogG3 + dlogG4)
    dlogZ = 1/6 * (dlogZ1 + 2 * dlogZ2 + 2 * dlogZ3 + dlogZ4)
    
    return dV, dlogG, dlogZ, dlogx

@njit
def solve_euler_equations(x, alpha, beta, Vhot, u, GAMMA, Nx):
    #desired level of accuracy
    TOL    = 1e-3
    
    G1 = (GAMMA + u) / (GAMMA - u)
    V1 = 2 * u / (u + GAMMA)
    Z1 = GAMMA * (1 + u) * (GAMMA - u) / (GAMMA + u)**2
    W1 = 1 - V1
    
    G     = zeros(Nx)
    G[-1] = G1
    V     = zeros(Nx)
    V[-1] = V1
    Z     = zeros(Nx)
    Z[-1] = Z1 

    #flag: 
    # 0 -- integrates down to zero
    # 1 -- V = 1 at xmin
    # 2 -- (1 - V)² approaches Z at xmin
    # 3 -- V becomes negative at xmin
    flag = 0
    xmin = 0.0

    #now integrate down from x = 1
    for i in range(Nx-1, 0, -1):
        #propagate to next step
        xnow  = x[i]
        xnext = x[i-1]
        Vnow  = V[i]
        Gnow  = G[i]
        Znow  = Z[i]
        
        dZ = 2 * TOL
        while xnow > xnext:
            dZ = absolute(Znow - (1 - Vnow)**2)
            if absolute(1 - Vnow) < TOL or dZ < TOL:
                break
        
            dlogx = absolute(xnext - xnow) / xnow
            #get gradients and timestep
            dV, dlogG, dlogZ, dlogx = RK4(xnow, dlogx, Vnow, Gnow, Znow, \
                                          GAMMA, beta, alpha)
            
            dVH0 = Vnow - (1 / beta / GAMMA \
                           + Vnow * (1 - Vnow) * (1 / alpha - Vnow) / 3 / Znow)
                
            Vnow -= dV
            Gnow *= (1 - dlogG)
            Znow *= (1 - dlogZ)
            xnow *= (1 - dlogx)
            
            Znow = Z1 * ((1 - Vnow) / W1)**(-1/beta) \
                      * (Gnow /G1)**((GAMMA-1) - 1/beta) \
                      * xnow**(-(2 + 3/beta))
                      
            dVH1 = Vnow - (1 / beta / GAMMA \
                           + Vnow * (1 - Vnow) * (1 / alpha - Vnow) / 3 / Znow)
            
            if dVH0 * dVH1 < 0:
                #sign change, set to VH
                Vnow = Vhot + Vnow * (1 - Vnow) * (1 / alpha - Vnow) / 3 / Znow 
        
        V[i-1] = Vnow
        G[i-1] = Gnow
        Z[i-1] = Znow

        if absolute(Vnow - 1) < TOL:
            xmin = xnow
            flag = 1
            return V, G, Z, xmin, flag
        elif dZ < TOL:
            xmin = xnow
            flag = 2
            return V, G, Z, xmin, flag
    #return results
    return V, G, Z, xmin, flag

def integration_wrapper(x, us, alphas, betas, Vhots, GAMMA, Nx, Nalpha, Nu):
    #setup output objects
    V = zeros((Nu, Nalpha, Nx))
    U = zeros((Nu, Nalpha, Nx))
    G = zeros((Nu, Nalpha, Nx))
    Z = zeros((Nu, Nalpha, Nx))
    P = zeros((Nu, Nalpha, Nx))
    T = zeros((Nu, Nalpha, Nx))
    
    xmin = zeros((Nu, Nalpha))
    Pc   = zeros((Nu, Nalpha))
    Tc   = zeros((Nu, Nalpha))
    flag = zeros((Nu, Nalpha))
    
    for i in range(Nu):
        u = us[i]
        for j in range(Nalpha):
            alpha = alphas[j]
            beta  = betas[j]
            Vhot  = Vhots[j]
            print("Start integration for alpha = %1.2f u = %1.2f"%(alpha, u))
            Vi, Gi, Zi, xmini, flagi = solve_euler_equations(x, alpha, beta, \
                                                             Vhot, u, GAMMA, \
                                                             Nx)
            xmin[i, j] = xmini
            flag[i, j] = flagi
            
            V[i, j] = Vi
            U[i, j] = Vi * x
            G[i, j] = Gi
            Z[i, j] = Zi
            T[i, j] = Zi * x**2 / GAMMA
            P[i, j] = Gi * T[i, j]
            
            #get critical values of pressure and Temperature
            if xmini > 0.0:
                cut = where(x >= xmin[i, j])
                Pc[i, j] = P[i, j][cut][0]
                Tc[i, j] = T[i, j][cut][0]
            else:
                #central value
                Pc[i, j] = P[i, j, 0]
                Tc[i, j] = T[i, j, 0]
    return V, U, G, Z, P, T, xmin, Pc, Tc, flag

def extend(x, V, U, G, Z, P, T, xmin, Pc, flag, alpha, alpha_c, beta, \
           Vhot, GAMMA, Nu, Nalpha, EXTEND_W, EXTEND_S):
    for i in range(Nu):
        for j in range(Nalpha):
            if EXTEND_W and flag[i, j] == 1:
                print("Extending solution for i = %d, alpha = %1.2f"%(i, alpha[j]))
                print("xc = %1.2f"%(xmin[i, j]))
                bub = where(x < xmin[i, j])
                shell = where(x >= xmin[i, j])
            
                Vc = V[i, j][shell][0]
            
                xw = x[bub] / xmin[i, j]
            
                #get mass in shell and mass in wind
                Mshell = trapz(x[shell], G[i, j][shell] * x[shell]**2)
                Mwind = max(1/3 - Mshell, 0.0)

                #isobaric
                P[i, j][bub] = Pc[i, j]
                U[i, j][bub] = xmin[i, j] * ((Vc - Vhot[j]) * xw**-2 + Vhot[j] * xw)
                V[i, j][bub] = U[i, j][bub] / x[bub]
                if Vhot[j] == 1:
                    G[i, j][bub] = exp(- Vhot[j] / (1 - Vc) * xw**3)
                else:
                    fc = (1 - Vhot[j]) / (Vc - Vhot[j])
                    G[i, j][bub] = (1 - fc * xw**3)**(Vhot[j] / (Vhot[j] - 1))
            
                #mass in wind
                norm = trapz(x[bub], G[i, j][bub] * x[bub]**2)
                G[i, j][bub] *= (Mwind / norm)
                
                #now compute Z and T
                T[i, j][bub] = P[i, j][bub] / (G[i, j][bub] + 1e-37)
                Z[i, j][bub] = GAMMA * T[i, j][bub] / x[bub]**2
            elif EXTEND_S and flag[i, j] == 2:
                print("Extending solution for i = %d, alpha = %1.2f"%(i, alpha[j]))
                print("xc = %1.2f"%(xmin[i, j]))
                bub = where(x < xmin[i, j])
                shell = where(x >= xmin[i, j])
            
                Vc = V[i, j][shell][0]
            
                xw = x[bub] / xmin[i, j]
            
                #get mass in shell and mass in wind
                Mshell = trapz(x[shell], G[i, j][shell] * x[shell]**2)
                Mwind = max(1/3 - Mshell, 0.0)
                
                #constant density
                G[i, j][bub] = 3 * Mwind / xmin[i, j]**3
                U[i, j][bub] = xmin[i, j] * Vc * xw**-2
                V[i, j][bub] = U[i, j][bub] / x[bub]
                Pnorm = P[i, j][shell][-1] * (G[i,j][shell][0] / G[i,j][shell][-1])**(GAMMA * (1 - Vhot[j]))
                P[i, j][bub] = ((1 - V[i, j][shell][-1]) / (xw**3 - Vc) / xmin[i, j]**3)**(GAMMA * Vhot[j])
                P[i, j][bub] *= Pnorm
            
                #now compute Z and T
                T[i, j][bub] = P[i, j][bub] / (G[i, j][bub] + 1e-37)
                Z[i, j][bub] = GAMMA * T[i, j][bub] / x[bub]**2
            
    return V, U, G, Z, P, T
            
            
            
    