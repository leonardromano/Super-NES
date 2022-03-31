#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:01:32 2022

@author: leonard
"""
#imports
#external libraries
import matplotlib.pylab as pl
from numpy import linspace
from time import time

#local libraries
import src.Euler_solver_1D as main
import src.plotting as plot

# User Interface
GAMMA  = 5/3    #adiabatic index

#define spatial range
minx = 0.01
Nx   = 1000

#define range of shock strengths
umin = 0.05
umax = 1
Nu   = 5

#define range of  (R ~ t^alpha)
alpha_min = 0.01
alpha_max = 1
Nalpha    = 1000

# Extend stellar winds to the center?
# Extend solution to center assuming constant Pressure (only SW)
EXTEND_W = False

# Extend single explosions to the center?
# Extend solution to center assuming constant density (only Single SNe)
EXTEND_S = False

#What do we analyze?
CRIT_XI  = True             # show critical xi as function of alpha
SOLUTION = False            # show numerical solution as function of xi

#colorscale for plots
c_alpha  = pl.cm.jet(linspace(0, 1, Nalpha))
c_u      = pl.cm.copper(linspace(0, 1, Nu))

###########################################################################
#code part

#compile
main.compile_funcs()

#setup u and alpha
u     = linspace(umin, umax, Nu)
alpha = linspace(alpha_min, alpha_max, Nalpha) #R ~ t^alpha
x     = linspace(minx, 1, Nx)

#get some constants
alpha_c = (1 + 2 * u) * (GAMMA + u) \
        / ((1 + 2 * u) * (GAMMA + u) + u * (2 * GAMMA + (3 * GAMMA + 1) * u))

beta = 3 * alpha / 2 / (1 - alpha)
Vhot = 1 / beta / GAMMA 

#get solutions (may take a while -> Time it)
t0 = time()
#all arrays have 2 (+1) dimensions
# 0 -- Nu entries for each value of u
# 1 -- Nalpha entries for each value of alpha
# 2 -- Nx entries for each x coordinate (only V, U, G, Z, P, T)
# V: Dimensionless velocity in units of local expansion speed xi*Vs
# U: Dimensionless velocity in units of shock speed Vs
# G: Density in units of ISM density
# Z: Square of sound-speed in units of local expansion speed xi*Vs
# P: Pressure in units of ISM density times shock speed squared
# T: Temperature in units of shock speed squared
# xmin: critical value of xi (where solution becomes singular)
# Pc: Dimensionless pressure at singularity
# Tc: Dimensionsless temperature at singularity
#flag: type of singularity at xmin
# 0 -- no singularity
# 1 -- V = 1
# 2 -- (1 - V)² approaches Z
V, U, G, Z, P, T, \
xmin, Pc, Tc, flag = main.integration_wrapper(x, u, alpha, beta, Vhot, \
                                              GAMMA, Nx, Nalpha, Nu)

#do we extend any of the solutions
EXTENDED = EXTEND_W | EXTEND_S

if EXTENDED:
    V, U, G, Z, P, T = main.extend(x, V, U, G, Z, P, T, xmin, Pc, \
                                   flag, alpha,  alpha_c, beta, Vhot, \
                                   GAMMA, Nu, Nalpha, EXTEND_W, EXTEND_S)
 
t1 = time()
print("Finished integrating 1D Euler equations for %d sets of parameters."%(Nalpha * Nu))
dt = t1 - t0
print("Took %g seconds (%g s / model"%(dt, dt / (Nalpha * Nu)))

if CRIT_XI:
    plot.critical_xi(alpha, xmin, flag, alpha_c, c_u, Nu)

if SOLUTION:
    plot.solution(x, U, G, P, T, xmin, alpha, u, Nalpha, Nu, EXTENDED)
