'''
An extension of the premlinary matrix reconstruction experiment which looks at the reconstruction NMF creates when the
ORIGINAL, 'TRUE' W matrix contains orthogonal basis vectors; the algorithm recovers a near-perfect reconstruction of the
resulting V_test matrix, but using alternative, non-orthogonal basis vectors spanning the same subspace. This clearly
shows the non-uniqueness of the decompositions NMF produces, but this is not problem in our application - see write up
for full discussion
'''

import numpy as np
import NMF_divergence as pc

folder_for_dummy_W_and_H = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_W_and_H\\"
folder_for_initializations = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_matrix_initializations\\"

n = 4
m = 4
r = 2


# A W matrix featuring orthogonal basis vectors
W_dummy = np.array([[1., 0.],
                    [1., 0.],
                    [0., 1.],
                    [0., 1.]])

# A randomly generated H matrix (checked that no ratios between contributions from two basis vectors are same)
H_dummy = np.array([[ 8.,  2.,  2.,  6.],
                    [ 5.,  4.,  8.,  3.]])

V_dummy = np.matmul(W_dummy, H_dummy)

# Random initializations
W_init = np.load(folder_for_initializations + 'dummy_W_init_n=4_m=4_r=2.npy')
H_init = np.load(folder_for_initializations + 'dummy_H_init_n=4_m=4_r=2.npy')


W_reconstructed, H_reconstructed, divergence_record = pc.NMF_divergence(V_dummy, W_init, H_init, n, m, r, 10000, 10)
V_reconstructed = np.matmul(W_reconstructed, H_reconstructed)

# Reconstruction residuals
print('TRUE V\n', V_dummy, '\n')
print('RECOVERED V\n', V_reconstructed, '\n')

# Max reconstruction error by element
max_residual = np.max(np.abs(V_reconstructed - V_dummy))
print('\nMax reconstruction residual of all elements: {}\n'.format(max_residual))


print('TRUE W\n', W_dummy, '\n')
print('RECOVERED W\n', W_reconstructed, '\n')
print('TRUE H\n', H_dummy, '\n')
print('RECOVERED H\n', H_reconstructed, '\n')

print('DOT PRODUCT BETWEEN RECOVERED BASIS VECTORS: ', np.dot(W_reconstructed[:,0], W_reconstructed[:,1]))

