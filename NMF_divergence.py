import pandas as pd
import numpy as np
import time
import os
import inspect

# NMF notation: V ~ WH where V is n x m, W is n x r, H is r x m
# Implementing divergence-based multiplicative updates algorithm from Lee and Seung, Algorithms for Non-negative Matrix Factorization (2001)

# A function to prevent zero-division errors by replacing anything rounded to zero with the smallest number numpy can handle

min_no = np.finfo(np.float64).eps

def make_safe(M):
    M[M == 0] = min_no
    return M


# Defining divergence loss (not strictly KL, but reduces to KL when matrices normalised to valid probability distribution - see Lee and Seung)

def D(V_real, V_approx):  # V_true, V_approx are any two matrices (np arrays) of the same dimensions

    V_real_dim = V_real.shape
    V_approx_dim = V_approx.shape
    assert V_real_dim == V_approx_dim, "Input matrix dimensions do not match: {} vs {}".format(V_real_dim, V_approx_dim)

    V_real_safe = make_safe(V_real)
    V_approx_safe = make_safe(V_approx)

    # Log of division is re-expressed as difference of logs to avoid overflows from division
    divergence = np.sum(
        np.multiply(V_real_safe, np.log(V_real_safe) - np.log(V_approx_safe)) - V_real_safe + V_approx_safe)

    return divergence


# Defining divergence update rules from Lee and Seung 2001

def W_update(Vw, Ww, Hw, n, m, r):
    assert Vw.shape == (n, m) and Ww.shape == (n, r) and Hw.shape == (r, m), "Matrix dimensions wrong"

    WHw = make_safe(np.matmul(Ww, Hw))
    divw = np.divide(Vw, WHw)

    numeratorw = np.matmul(divw, np.transpose(Hw))
    denominatorw = np.matmul(np.ones((n, m)), np.transpose(Hw))

    denominatorw = make_safe(denominatorw)
    fractionw = np.divide(numeratorw, denominatorw)

    return np.multiply(Ww, fractionw)


def H_update(Vh, Wh, Hh, n, m, r):
    assert Vh.shape == (n, m) and Wh.shape == (n, r) and Hh.shape == (r, m), "Matrix dimensions wrong"

    WHh = make_safe(np.matmul(Wh, Hh))
    divh = np.divide(Vh, WHh)

    numeratorh = np.matmul(np.transpose(Wh), divh)
    denominatorh = np.matmul(np.transpose(Wh), np.ones((n, m)))

    denominatorh = make_safe(denominatorh)
    fraction = np.divide(numeratorh, denominatorh)

    return np.multiply(Hh, fraction)


from matplotlib import pyplot as plt


# Carrying out NMF

def NMF_divergence(V_true, W_init, H_init, n, m, r, max_iterations, record_D_every_x_iterations=10,
                   report_progress=False, save_progress_to=None):
    W = W_init
    H = H_init
    del W_init, H_init

    divergence_by_it = []

    for it in range(max_iterations):

        if it % record_D_every_x_iterations == 0:

            V_reconstruct = np.matmul(W, H)

            div = D(V_true.copy(), V_reconstruct)

            divergence_by_it.append([it, div])

            if save_progress_to != None:

                # if count % 10 == 0:
                # H_10 = np.zeros([10, r, m])
                np.save(save_progress_to + 'H_r={}_it={}.npy'.format(r, it), H)
                if (it / record_D_every_x_iterations) % 10 == 0:
                    np.save(save_progress_to + 'W_r={}_it={}.npy'.format(r, it), W[:, :3])

            start_time = time.time()

            W = W_update(V_true, W, H, n, m, r)
            H = H_update(V_true, W, H, n, m, r)

            time_for_it = time.time() - start_time

            if report_progress:
                print('Rank {}, Iteration {}/{} complete, divergence = {}'.format(r, it + 1, max_iterations, div))
                print('Time taken for this iteration: {}s'.format(time_for_it), '\n')

        else:

            W = W_update(V_true, W, H, n, m, r)
            H = H_update(V_true, W, H, n, m, r)

    divergence_by_it = pd.DataFrame(divergence_by_it).set_index(0)
    return W, H, divergence_by_it


# A function to format and save results to CSVs

def save_results(main_results_folder, W, H, unique_name='', additional_data_name_to_array_dict={},
                 row_names_list=None, column_names_list=None):
    if unique_name == '':
        timestr = time.strftime("%Y%m%d-%H%M%S")
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        scriptname = mod.__file__
        file_prefix = scriptname[scriptname.rfind('/') + 1:scriptname.rfind('.')]
        output_data_folder_name = '{}_output-{}/'.format(file_prefix, timestr)
        output_data_folder_path = main_results_folder + output_data_folder_name
        os.mkdir(output_data_folder_path)

    else:
        output_data_folder_path = main_results_folder

    print('Saving output to {} ...'.format(output_data_folder_path))

    assert H.shape[0] == W.shape[1], 'W and H dimensions do not match'

    if row_names_list != None and column_names_list != None:
        W = pd.DataFrame(W, index=row_names_list)
        H = pd.DataFrame(H, columns=column_names_list)
    else:
        W = pd.DataFrame(W)
        H = pd.DataFrame(H)

    W.to_csv(output_data_folder_path + 'W_{}.csv'.format(unique_name))
    H.to_csv(output_data_folder_path + 'H_{}.csv'.format(unique_name))

    frame_ref = 0
    for name, array in additional_data_name_to_array_dict.items():
        frame = pd.DataFrame(array)
        frame.to_csv(output_data_folder_path + name + '_{}.csv'.format(unique_name))
        frame_ref += 1

    print('All results saved.')