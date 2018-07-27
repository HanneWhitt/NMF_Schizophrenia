# A script to generate and save random matrices of integers W_test and H_test and random initialisations W_init and H_init for test of NMF code
# See matrix_reconstruction_experiment.py for use

import numpy as np
import os

folder_for_dummy_W_and_H = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_W_and_H"
folder_for_dummy_V = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_V"
folder_for_initializations = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_matrix_initializations"
n_m_max = 10
max_int = 10

def random_W_and_H(n, m, r, max_integer):
    W_test = np.random.randint(0, max_integer, (n, r)).astype(float)
    H_test = np.random.randint(0, max_integer, (r, m)).astype(float)
    return W_test, H_test

def random_V(n, m, r, max_integer):
    return np.random.randint(0, max_integer, (n, m)).astype(float)

def random_initialization(n, m, r):
    W_init = np.random.uniform(0.0, 1.0, size=(n, r))
    H_init = np.random.uniform(0.0, 1.0, size=(r, m))
    return W_init, H_init


for folder in [folder_for_dummy_W_and_H, folder_for_dummy_V, folder_for_initializations]:
    os.mkdir(folder)

for n in range(2, n_m_max + 1):
    for m in range(2, n_m_max + 1):
        for r in range(2, max([n, m])):

            print('n = {}, m = {}, r = {}...'.format(n, m, r))

            W_test, H_test = random_W_and_H(n, m, r, max_int)
            np.save('{}\\dummy_W_test_n={}_m={}_r={}'.format(folder_for_dummy_W_and_H, n, m, r), W_test)
            np.save('{}\\dummy_H_test_n={}_m={}_r={}'.format(folder_for_dummy_W_and_H, n, m, r), H_test)

            V_test = random_V(n, m, r, max_int)
            np.save('{}\\dummy_V_test_n={}_m={}_r={}'.format(folder_for_dummy_V, n, m, r), V_test)

            W_init, H_init = random_initialization(n, m, r)
            np.save('{}\\dummy_W_init_n={}_m={}_r={}'.format(folder_for_initializations, n, m, r), W_init)
            np.save('{}\\dummy_H_init_n={}_m={}_r={}'.format(folder_for_initializations, n, m, r), H_init)

            print('Saved\n')