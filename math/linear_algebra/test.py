#!/usr/bin/env python3

import numpy as np
import time
cat_matrices = __import__('102-squashed_like_sardines').cat_matrices

mat1 = [1, 2, 3]
mat2 = [4, 5, 6]
np_mat1 = np.array(mat1)
np_mat2 = np.array(mat2)

t0 = time.time()
m = cat_matrices(mat1, mat2)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat1, np_mat2))
t1 = time.time()
print(t1 - t0, "\n")

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
np_mat1 = np.array(mat1)
np_mat2 = np.array(mat2)

t0 = time.time()
m = cat_matrices(mat1, mat2)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((np_mat1, np_mat2))
t1 = time.time()
print(t1 - t0, "\n")

t0 = time.time()
m = cat_matrices(mat1, mat2, axis=1)
t1 = time.time()
print(t1 - t0)
print(m)
t0 = time.time()
np.concatenate((mat1, mat2), axis=1)
t1 = time.time()
print(t1 - t0, "\n")
