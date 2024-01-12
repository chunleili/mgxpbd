import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import scipy.sparse.csgraph as csgraph
import os
def test():
    # read from text
    os.chdir('Build/Debug')
    print('cwd: ',os.getcwd())
    coo_i_arr = np.loadtxt('coo_i_arr.txt', dtype=int)
    coo_j_arr = np.loadtxt('coo_j_arr.txt', dtype=int)
    coo_v_arr = np.loadtxt('coo_v_arr.txt', dtype=float)

    csr_row_start = np.loadtxt('csr_row_start.txt', dtype=int)
    csr_col_idx = np.loadtxt('csr_col_idx.txt', dtype=int)
    csr_val = np.loadtxt('csr_val.txt', dtype=float)

    mat_coo = sp.coo_array((coo_v_arr, (coo_i_arr, coo_j_arr)))
    mat_csr = sp.csr_array((csr_val, csr_col_idx, csr_row_start))
    
    mat_coo = mat_coo.tocsr()
    # # 使用重新排序后的索引对稀疏矩阵进行重排
    # perm = csgraph.reverse_cuthill_mckee(mat_coo)
    # reordered_coo = mat_coo[perm, :][:, perm]
    # perm = csgraph.reverse_cuthill_mckee(mat_csr)
    # reordered_csr = mat_csr[perm, :][:, perm]

    mmwrite( "coo.mtx", mat_coo)
    mmwrite( "csr.mtx", mat_csr)


        

if __name__ == '__main__':
    test()