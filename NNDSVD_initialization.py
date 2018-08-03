'''
Non-negative Double Singular Value Decomposition is a popular initialisation method for NMF, which results in faster
convergence to a better loss value. The code below was implemented with reference to Boutsidis and Gallopoulos (2007),
"SVD based initialization: A head start for nonnegative matrix factorization". See write up for more details

There are three sub-types mentioned in the paper, all of which are implemented as functions here.
1) Standard NNDSVD - produces a matrix typically containing many zeros. This is a problem given that we use the
multiplicative updates algorithm, which cannot update zeros. Hence we only use the following two variations.
2) NNDSVDa - Carry out NNDSVD, then replace the zero elements with the average of the elements of the matrix to be
decomposed (X_average)
3) NNDSVDar - Carry out NNDSVD, then replace the zero elements with uniform random numbers from the range:
[0, X_average/100]

Methods 2 and three will be used in our study to attempt to find better solutions for less iterations.

'''

import numpy as np


def positive_part(a):
    return np.array([a >= 0])*a

def negative_part(a):
    return np.array([a < 0]) * -a


def NNDSVD_initialization(X, n, m, r, report_progress = True):

    # Singular Value Decomposition
    if report_progress:
        print('\n\nNNDSVD INITIALISATION\n\nCalculating SVD...')

    U, S, VT = np.linalg.svd(X)

    W_init = np.zeros((n, r))
    H_init = np.zeros((r, m))

    if report_progress:
        print('Complete. Initialising column 1...')

    # Initialising first columns using first singular triplet from SVD of full matrix
    W_init[:, 0] = np.sqrt(S[0])*np.abs(U[:,0])
    H_init[0, :] = np.sqrt(S[0])*np.abs(VT[0, :])

    # Initialising remaining columns one at a time using a second SVD
    for j in range(1, r):

        if report_progress:
            print('Initialising column {}...'.format(j + 1))

        u_j = U[:, j]
        v_j = VT[j, :]

        u_j_pos = positive_part(u_j)
        u_j_neg = negative_part(u_j)
        v_j_pos = positive_part(v_j)
        v_j_neg = negative_part(v_j)

        magnitude_u_j_pos = np.linalg.norm(u_j_pos)
        magnitude_v_j_pos = np.linalg.norm(v_j_pos)
        pos_mag_product = magnitude_u_j_pos*magnitude_v_j_pos

        magnitude_u_j_neg = np.linalg.norm(u_j_neg)
        magnitude_v_j_neg = np.linalg.norm(v_j_neg)
        neg_mag_product = magnitude_u_j_neg*magnitude_v_j_neg

        if pos_mag_product > neg_mag_product:
            W_init_col = u_j_pos / magnitude_u_j_pos
            H_init_col = v_j_pos/magnitude_v_j_pos
            sigma = pos_mag_product

        else:
            W_init_col = u_j_neg / magnitude_u_j_neg
            H_init_col = v_j_neg/magnitude_v_j_neg
            sigma = neg_mag_product

        W_init[:, j] = np.sqrt(S[j] * sigma) * W_init_col
        H_init[j, :] = np.sqrt(S[j] * sigma) * H_init_col

    if report_progress:
        print('\nCOMPLETE')

    return W_init, H_init



def NNDSVDa_initialization(X, n, m, r, report_progress = True):

    W_init, H_init = NNDSVD_initialization(X, n, m, r, report_progress)

    if report_progress:
        print('\nNNDSVDa - replacing zeroes with average')

    X_average = np.average(X)

    W_init = W_init + np.array([W_init == 0]) * X_average
    H_init = H_init + np.array([H_init == 0]) * X_average

    if report_progress:
        print('\nCOMPLETE')

    return W_init.reshape((n, r)), H_init.reshape((r, m))



def NNDSVDar_initialization(X, n, m, r, report_progress = True, random_seed = 42):

    W_init, H_init = NNDSVD_initialization(X, n, m, r, report_progress)

    if report_progress:
        print('\nNNDSVDar - replacing zeroes with uniform randoms in range [0, AVERAGE/100]')

    X_average = np.average(X)

    random_state = np.random.RandomState(random_seed)
    W_rand = random_state.uniform(0.0, X_average/100, (n, r))
    H_rand = random_state.uniform(0.0, X_average/100, (r, m))

    W_init = W_init + np.array([W_init == 0]) * W_rand
    H_init = H_init + np.array([H_init == 0]) * H_rand

    if report_progress:
        print('\nCOMPLETE')

    return W_init.reshape((n, r)), H_init.reshape((r, m))





# Testing on some previously studied matrices of integers

if __name__ == "__main__":

    # A W matrix featuring orthogonal basis vectors
    W_dummy = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.]])

    # A randomly generated H matrix (checked that no ratios between contributions from two basis vectors are same)
    H_dummy = np.array([[ 8.,  2.,  2.,  6.],
                        [ 5.,  4.,  8.,  3.]])

    V_dummy = np.matmul(W_dummy, H_dummy)

    print('V_dummy')
    print(V_dummy)

    W_init, H_init = NNDSVD_initialization(V_dummy, 4, 4, 2)
    W_init_a, H_init_a = NNDSVDa_initialization(V_dummy, 4, 4, 2)
    W_init_ar, H_init_ar = NNDSVDar_initialization(V_dummy, 4, 4, 2)

    print('NNDSVD Initialization W')
    print(W_init)
    print(W_init.shape)
    print('NNDSVD Initialization H')
    print(H_init)
    print(H_init.shape)

    print('NNDSVDa Initialization W')
    print(W_init_a)
    print(W_init_a.shape)

    print('NNDSVDa Initialization H')
    print(H_init_a)
    print(H_init_a.shape)

    print('NNDSVDar Initialization W')
    print(W_init_ar)
    print(W_init_ar.shape)

    print('NNDSVDar Initialization H')
    print(H_init_ar)
    print(H_init_ar.shape)


    # A randomly generated W matrix
    W_dummy = np.load("C:/Users\hanne\Documents\PROJECT\Project "
                      "Data\dummy_matrix_initializations\dummy_W_init_n=10_m=10_r=9.npy")

    # A randomly generated H matrix
    H_dummy = np.load("C:/Users\hanne\Documents\PROJECT\Project "
                      "Data\dummy_matrix_initializations\dummy_H_init_n=10_m=10_r=9.npy")


    V_dummy = np.matmul(W_dummy, H_dummy)

    W_init, H_init = NNDSVD_initialization(V_dummy,  10, 10, 9)
    W_init_a, H_init_a = NNDSVDa_initialization(V_dummy,  10, 10, 9)
    W_init_ar, H_init_ar = NNDSVDar_initialization(V_dummy,  10, 10, 9)

    print('NNDSVD Initialization W')
    print(W_init)
    print(W_init.shape)
    print('NNDSVD Initialization H')
    print(H_init)
    print(H_init.shape)

    print('NNDSVDa Initialization W')
    print(W_init_a)
    print(W_init_a.shape)

    print('NNDSVDa Initialization H')
    print(H_init_a)
    print(H_init_a.shape)

    print('NNDSVDar Initialization W')
    print(W_init_ar)
    print(W_init_ar.shape)

    print('NNDSVDar Initialization H')
    print(H_init_ar)
    print(H_init_ar.shape)