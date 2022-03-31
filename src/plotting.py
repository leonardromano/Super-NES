#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:57:49 2022

@author: leonard
"""
from numpy import where
import matplotlib.pyplot as plt

from os import path, makedirs

#plot critical xi
def critical_xi(alpha, xmin, flag, alpha_c, c_u, Nu):
    #check if output directories exist and if not make them
    if not path.exists("figures/"):
        makedirs("figures/")
    
    for i in range(Nu):
        V1 = where(flag[i] == 1)
        VZ = where(flag[i] == 2)
        plt.plot(alpha, xmin[i], color = c_u[i])
        #plot where singularity changes
        plt.plot(alpha[V1[0][0]], xmin[i, V1[0][0]], color = "blue", \
                 ls = "", marker="o", markersize = 5)
        plt.plot(alpha[V1[0][-1]], xmin[i, V1[0][-1]], color = "blue", \
                 ls = "", marker="o", markersize = 5)
            
        plt.plot(alpha[VZ[0][0]], xmin[i, VZ[0][0]], color = "red", \
                 ls = "", marker="o", markersize = 5)
        plt.plot(alpha[VZ[0][-1]], xmin[i, VZ[0][-1]], color = "red", \
                 ls = "", marker="o", markersize = 5)
        
    plt.xlabel(r"$\alpha = dlogR/dlogt$")
    plt.ylabel(r"$\xi_c$")
    plt.xlim(0, 1)
    plt.savefig("figures/critical_xi_full.pdf", bbox_inches='tight')
    plt.show()

    #plot critical xi (only simple explosion)
    for i in range(Nu):
        SE = where(alpha < alpha_c[i])
        plt.plot(alpha[SE], xmin[i][SE], color = c_u[i])

    plt.xlabel(r"$\alpha = dlogR/dlogt$")
    plt.ylabel(r"$\xi_c$")
    plt.xlim(0, 1)
    plt.savefig("figures/critical_xi_SE.pdf", bbox_inches='tight')
    plt.show()

    #plot critical xi (only stellar winds)
    for i in range(Nu):
        SW = where(alpha >= alpha_c[i])
        plt.plot(alpha[SW], xmin[i][SW], color = c_u[i])

    plt.xlabel(r"$\alpha = dlogR/dlogt$")
    plt.ylabel(r"$\xi_c$")
    plt.xlim(0, 1)
    plt.savefig("figures/critical_xi_SW.pdf", bbox_inches='tight')
    plt.show()
    
def solution(x, U, G, P, T, xmin, alpha, u, Nalpha, Nu, EXTENDED):
    #check if output directories exist and if not make them
    if not path.exists("figures/solution/"):
        makedirs("figures/solution/")
        
    for i in range(Nu):
        for j in range(Nalpha):
            cut = where(P[i, j] > 0.0)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            
            ax.plot(x[cut], U[i, j][cut] / U[i, j][-1], color = "blue", \
                     label = "v/v(1)")
            ax.plot(x[cut], G[i, j][cut] / G[i, j][-1], color = "brown", \
                     label = "G / G(1)", ls="--")
            ax.plot(x[cut], P[i, j][cut] / P[i, j][-1], color = "brown", \
                     label = "P / P(1)", ls=":")
            ax2.plot(x[cut], T[i, j][cut] / T[i, j][-1], color = "green", \
                     label="T/T(1)")
                 
            ax.set_xlabel(r"$\xi$")
            ax.set_ylabel(r"$v/v_{1}, G/G_{1}, P/P_{1}$")
            ax.set_xlim(x[cut[0][0]], 1)
            if EXTENDED:
                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.vlines(xmin[i, j], *ax.get_ylim(), linestyle="--", \
                          color = "black")
            
            ax2.set_ylabel(r"$T/T_{1}$")
            ax2.set_yscale("log")
            ax.legend(loc = "best", fontsize = 12)
            fig.savefig("figures/solution/u%1.2f_alpha%1.2f.pdf"%(u[i], alpha[j]), \
                        bbox_inches='tight')
            fig.clear()
            plt.close(fig)
            
                
            
            