import pandas as pd
import numpy as np
from NMF_divergence import D, NMF_divergence, save_results

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/"
initializations_path = data_path + 'Experiment_1_initializations/'
results_path_main = data_path + 'Experiment_1_results/'
results_path_permuted = data_path + 'Experiment_1_results_permuted/'


V_15000 = pd.read_csv(data_path + 'Experiment_1_Data.csv', index_col = 0)
V_15000_columns_permuted = pd.read_csv(data_path + 'Experiment_1_Data_columns_permuted.csv', index_col = 0)

patients = list(V_15000.columns)
genes = list(V_15000.index)

V_15000 = np.array(V_15000)
V_15000_columns_permuted = np.array(V_15000_columns_permuted)


def random_initialization(n, m, r):
    W_init = np.random.uniform(0.0, 1.0, size=(n, r))
    H_init = np.random.uniform(0.0, 1.0, size=(r, m))
    return W_init, H_init


n = V_15000.shape[0]
m = V_15000.shape[1]

iterations = 100
divergence_calc_frequency = 1



# Ranks to evaluate in experiment 1
ranks = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]

for repeat in range(20):

    for r in ranks:

        code = 'rep={}_rank={}'.format(repeat, r)
        print(code)

        # generating initializations
        W_init, H_init = random_initialization(n, m, r)
        np.save(initializations_path + 'W_init_' + code, W_init)
        np.save(initializations_path + 'H_init_' + code, H_init)

        # REAL NMF
        W, H, divergence_by_it = NMF_divergence(V_15000, W_init.copy(), H_init.copy(), n, m, r, iterations, divergence_calc_frequency, report_progress=True)
        save_results(W, H, {'divergence_record': divergence_by_it}, row_names_list=genes, column_names_list=patients,
                     main_results_folder=results_path_main)

        # PERMUTATION NMF
        W, H, divergence_by_it = NMF_divergence(V_15000_columns_permuted, W_init, H_init, n, m, r, iterations, divergence_calc_frequency, report_progress=True)
        save_results(W, H, {'divergence_record': divergence_by_it}, row_names_list=genes, column_names_list=patients,
                     main_results_folder=results_path_permuted)