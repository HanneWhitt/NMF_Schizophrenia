# This script runs an experiment to validate the constructed code, by testing its ability to reconstruct randomly
# generated matrices. Also compares performance of project code to sci-kit learn's NMF implementation
# See write up for full description


import numpy as np
import pandas as pd
from NMF_divergence import NMF_divergence as project_code_NMF
from sklearn.decomposition import NMF as sklearn_NMF
import matplotlib.pyplot as plt
import time


folder_for_dummy_W_and_H = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_W_and_H"
folder_for_initializations = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy_matrix_initializations"
results_folder = "C:\\Users\\hanne\\Documents\\PROJECT\\Project Data\\dummy matrix test results\\"
n_m_max = 10
max_iterations = 1000


def get_error_statistics(V_true, V_reconstruct):

    # Absoluter reconstruction errors
    recons_errors = np.abs(V_reconstruct - V_true)

    # if reconstruction errors for all elements are zero
    if not np.any(recons_errors):
        return 0.0, 0.0, 0.0, 0.0

    else:

        average_absolute_error = np.average(recons_errors)
        max_absolute_error = np.max(recons_errors)

        recons_errors_safe = recons_errors[V_true != 0]
        V_true_safe = V_true[V_true != 0]

        relative_errors = recons_errors_safe / V_true_safe

        average_relative_error = np.average(relative_errors)
        max_relative_error = np.max(relative_errors)

        return average_absolute_error, max_absolute_error, average_relative_error, max_relative_error


def plot_divergence_by_iteration(divergence_vector, title):
    plt.plot(divergence_vector)
    plt.title(title)
    plt.xlabel('Iteration number')
    plt.ylabel('Divergence')
    div_start = divergence_vector[0]
    div_end = divergence_vector[-1]
    text_x = 200
    text_y = div_end + 0.05*(div_start - div_end)
    plt.text(text_x, text_y, 'Final Div.: {}'.format(div_end))
    plt.show()


error_stats_columns = ['MEAN ABS PC', 'MEAN ABS SK', 'MAX ABS PC', 'MAX ABS SK', 'MEAN REL PC', 'MEAN REL SK',
                       'MAX REL PC', 'MAX REL SK', 'TIME PC', 'TIME SK', 'NO. ELEMENTS']


def check_NMF_reconstruction(V_test, W_init, H_init, n, m, r, iterations):

    # Carrying out NMF using project code
    pc_start = time.time()
    W_pc, H_pc, convergence_record = project_code_NMF(V_test, W_init, H_init, n, m, r, iterations,
                                                      record_D_every_x_iterations = 10)
    pc_time = time.time() - pc_start
    V_pc = np.matmul(W_pc, H_pc)

    # Carrying out NMF using scikit-learn
    sk_start = time.time()
    model = sklearn_NMF(n_components=r, max_iter=iterations, tol=0, verbose=False, init='custom', solver='mu',
                        beta_loss='kullback-leibler')
    W_sklearn = model.fit_transform(V_test, W=W_init, H=H_init)
    H_sklearn = model.components_
    sk_time = time.time() - sk_start
    V_sklearn = np.matmul(W_sklearn, H_sklearn)

    # Calculating/viewing reconstruction error statistics
    pc_av_abs, pc_max_abs, pc_av_rel, pc_max_rel = get_error_statistics(V_test, V_pc)
    sk_av_abs, sk_max_abs, sk_av_rel, sk_max_rel = get_error_statistics(V_test, V_sklearn)

    no_elements = n*m

    results = pd.DataFrame([[pc_av_abs, sk_av_abs, pc_max_abs, sk_max_abs, pc_av_rel, sk_av_rel, pc_max_rel, sk_max_rel,
                             pc_time, sk_time, no_elements]],
                           columns = error_stats_columns,
                           index = ['n={}, m={}, r={}'.format(n, m, r)])

    return results


full_results = pd.DataFrame(columns=error_stats_columns)

for n in range(2, n_m_max + 1):
    for m in range(2, n_m_max + 1):
        for r in range(2, min([n, m])):


            print('n = {}, m = {}, r = {}...'.format(n, m, r))

            W_init = np.load('{}\\dummy_W_init_n={}_m={}_r={}.npy'.format(folder_for_initializations, n, m, r))
            H_init = np.load('{}\\dummy_H_init_n={}_m={}_r={}.npy'.format(folder_for_initializations, n, m, r))

            W_test = np.load('{}\\dummy_W_test_n={}_m={}_r={}.npy'.format(folder_for_dummy_W_and_H, n, m, r))
            H_test = np.load('{}\\dummy_H_test_n={}_m={}_r={}.npy'.format(folder_for_dummy_W_and_H, n, m, r))

            V_test = np.matmul(W_test, H_test)

            results = check_NMF_reconstruction(V_test, W_init, H_init, n, m, r, max_iterations)

            full_results = full_results.append(results)


#full_results.to_csv(results_folder + 'full_results.csv')

overall_average_relative_reconstruction_error_pc = full_results['MEAN REL PC'].mean()
print('overall_average_relative_reconstruction_error_pc ', overall_average_relative_reconstruction_error_pc)

max_average_relative_reconstruction_error_pc = full_results['MEAN REL PC'].max()
print('max_average_relative_reconstruction_error_pc ', max_average_relative_reconstruction_error_pc)

pc_outperforms = np.sum(np.where(full_results['MEAN REL PC'] < full_results['MEAN REL SK'], 1, 0))
pc_matches = np.sum(np.where(full_results['MEAN REL PC'] == full_results['MEAN REL SK'], 1, 0))

pc_matches_or_outperforms_proportion = (pc_matches + pc_outperforms)/204
print('pc_matches_or_outperforms_proportion ', pc_matches_or_outperforms_proportion)

pc_outperforms_proportion = pc_outperforms/204
print('pc_outperforms_proportion ', pc_outperforms_proportion)

pc_total_time = np.sum(full_results['TIME PC'])
print('pc_total_time ', pc_total_time)

sk_total_time = np.sum(full_results['TIME SK'])
print('sk_total_time', sk_total_time)
