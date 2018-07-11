import pandas as pd
import numpy as np
import time
import os
import inspect

# NMF notation: V ~ WH where V is n x m, W is n x r, H is r x m
# Implemetning divergence-based multiplicative updates algorithm from Lee and Seung, Algorithms for Non-negative Matrix Factorization (2001)

# A function to prevent zero-division errors by replacing anything rounded to zero with the smallest number numpy can handle

min_no = np.finfo(np.float64).eps

def make_safe(M):
    M[M == 0] = min_no
    return M


# Defining divergence loss (not strictly KL, but reduces to KL when matrices normalised to valid probability distribution - see Lee and Seung)

def D(V_real, V_approx): # V_true, V_approx are any two matrices (np arrays) of the same dimensions

    V_real_dim = V_real.shape
    V_approx_dim = V_approx.shape
    assert V_real_dim == V_approx_dim, "Input matrix dimensions do not match"

    V_real_safe = make_safe(V_real)
    V_approx_safe = make_safe(V_approx)

    divergence = np.sum(np.multiply(V_real_safe, np.log(V_real_safe) - np.log(V_approx_safe)) - V_real_safe + V_approx_safe)

    return divergence


# Defining divergence update rules from Lee and Seung 2001

def W_update(Vw, Ww, Hw, n, m, r, print_progress = False):

    assert Vw.shape == (n, m) and Ww.shape == (n, r) and Hw.shape == (r, m), "Matrix dimensions wrong"

    WHw = make_safe(np.matmul(Ww, Hw))
    divw = np.divide(Vw, WHw)
    
    numeratorw = np.matmul(divw, np.transpose(Hw))
    denominatorw =  np.matmul(np.ones((n, m)), np.transpose(Hw))

    denominatorw = make_safe(denominatorw)
    fractionw = np.divide(numeratorw, denominatorw)

    return np.multiply(Ww, fractionw)


def H_update(Vh, Wh, Hh, n, m, r, print_progress = False):

    assert Vh.shape == (n, m) and Wh.shape == (n, r) and Hh.shape == (r, m), "Matrix dimensions wrong"

    WHh = make_safe(np.matmul(Wh, Hh))
    divh = np.divide(Vh, WHh)

    numeratorh = np.matmul(np.transpose(Wh), divh)
    denominatorh = np.matmul(np.transpose(Wh), np.ones((n, m)))

    denominatorh = make_safe(denominatorh)
    fraction = np.divide(numeratorh, denominatorh)

    return np.multiply(Hh, fraction)


# Carrying out NMF

def NMF_divergence(V_true, W_init, H_init, n, m, r, iterations, report_progress_every_x_iterations = None):

    W = W_init
    H = H_init
    del W_init, H_init

    divergence_by_it = []

    for it in range(iterations):

        start_time = time.time()

        W = W_update(V_true, W, H, n, m, r)
        H = H_update(V_true, W, H, n, m, r)

        V_reconstruct = np.matmul(W, H)

        div = D(V_true.copy(), V_reconstruct)
        sqrt_2_x_div = np.sqrt(div * 2)

        divergence_by_it.append((div, sqrt_2_x_div))

        time_for_iteration = time.time() - start_time

        if report_progress_every_x_iterations != None:
            if (it + 1) % report_progress_every_x_iterations == 0:
                print(
                    'Iteration {}/{} complete, divergence = {}, sqrt(2*divergence) = {}'.format(it + 1, iterations, div,
                                                                                                sqrt_2_x_div))
                print('Time taken: {}s'.format(time_for_iteration), '\n')

    divergence_by_it = np.array(divergence_by_it)

    return W, H, divergence_by_it


# A function to format and save results to CSVs

def save_results(W, H, additional_data_name_to_array_dict = {}, row_names_list = None, column_names_list = None, main_results_folder = None):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    scriptname = mod.__file__
    file_prefix = scriptname[scriptname.rfind('/') + 1:scriptname.rfind('.')]
    output_data_folder_name = '{}_output-{}'.format(file_prefix, timestr)

    if main_results_folder != None:
        output_data_folder_path = main_results_folder + '\\' + output_data_folder_name
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_data_folder_path = dir_path + '\\' + output_data_folder_name

    print('Saving output to {} ...'.format(output_data_folder_path))

    os.mkdir(output_data_folder_path)

    n = W.shape[0]
    m = H.shape[1]
    assert H.shape[0] == W.shape[1], 'W and H dimensions do not match'
    r = H.shape[0]

    if row_names_list != None and column_names_list != None:
        W = pd.DataFrame(W, index=row_names_list)
        H = pd.DataFrame(H, columns=column_names_list)
    else:
        W = pd.DataFrame(W)
        H = pd.DataFrame(H)

    W.to_csv(output_data_folder_path + '\\' + 'W_r' + str(r) + '.csv')
    H.to_csv(output_data_folder_path + '\\' + 'H_r' + str(r) + '.csv')

    frame_ref = 0
    for name, array in additional_data_name_to_array_dict.items():
        frame = pd.DataFrame(array)
        frame.to_csv(output_data_folder_path + '\\' + name + '.csv')
        frame_ref += 1

    print('All results saved.')