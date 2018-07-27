# Experiment 1b took the best repeats for each rank from experiment 1 and ran them for more iterations to achieve or get closer to convergence


import pandas as pd
import numpy as np
from NMF_divergence import D, NMF_divergence, save_results
from analysis_functions import results_path_by_index


data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/"

initializations_path = data_path + 'Experiment_1_initializations/'
results_path_main = data_path + 'Experiment_1b_results/'
#results_path_permuted = data_path + 'Experiment_1b_results_permuted/'


V_15000 = pd.read_csv(data_path + 'Experiment_1_Data.csv', index_col = 0)

patients = list(V_15000.columns)
genes = list(V_15000.index)

V_15000 = np.array(V_15000)


n = V_15000.shape[0]
m = V_15000.shape[1]

iterations = 1000
divergence_calc_frequency = 10



# Ranks to evaluate in experiment 1
ranks = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]

best_repeats = [10, 17, 10, 5, 14, 2, 17, 9, 15, 8, 18, 4, 15]


for idx in range(13):

    r = ranks[idx]
    repeat = best_repeats[idx]

    # Loading best initialisations by divergence after 100 iterations as initialisation for longer run

    code = 'rep={}_rank={}'.format(repeat, r)
    print(code)

    W_init = np.load(initializations_path + 'W_init_' + code + '.npy')
    H_init = np.load(initializations_path + 'H_init_' + code + '.npy')

    # REAL NMF
    W, H, divergence_by_it = NMF_divergence(V_15000, W_init.copy(), H_init.copy(), n, m, r, iterations,
                                            divergence_calc_frequency, report_progress=True)
    save_results(results_path_main, W, H, unique_name='rank={}'.format(r),
                 additional_data_name_to_array_dict={'divergence_record': divergence_by_it},
                 row_names_list=genes, column_names_list=patients)
