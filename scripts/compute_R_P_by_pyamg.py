''' generate R and P by pyamg'''
import pyamg
import numpy as np
import scipy
import os, sys

sys.path.append(os.getcwd())

A = scipy.io.mmread("./data/misc/A_100.mtx")
A = A.tocsr()
print("generate R and P by pyamg...")
ml = pyamg.classical.ruge_stuben_solver(A, max_levels=2)  # construct the multigrid hierarchy
P = ml.levels[0].P
R = ml.levels[0].R
print("generation done")
print("P shape: ", P.shape)
print("R shape: ", R.shape)
print("export R and P to data/misc/R.pyamg.mtx and data/misc/P.pyamg.mtx")
scipy.io.mmwrite("./data/misc/R_100.pyamg.mtx", R)
scipy.io.mmwrite("./data/misc/P_100.pyamg.mtx", P)