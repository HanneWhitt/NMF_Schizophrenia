# Experiment 2b repeats experiment 2 using NNDSVDa and NNDSVDar initialization instead of random. These results will
# be used to compare the speed of convergence.

# Calculating initialisation:
# NOTE that increasing rank of NNDSVD initialization simply adds new columns, leaving earlier columns unchanged. Hence,
# we can simply compute the initialisation for the maximum rank studied and then splice to get required rows and
# columns. This avoids repeating the expensive initial SVD computation.

from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np
from NNDSVD_initialization import NNDSVDa_initialization, NNDSVDar_initialization
import os

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/CM_experiment_2_data.csv"

results_folder_a = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDa/"
results_folder_ar = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDar/"

data_matrix = pd.read_csv(data_path, index_col=0)

genes = list(data_matrix.index)
patients = list(data_matrix.columns)

n = len(genes)
m = len(patients)

data_matrix = np.array(data_matrix)



def experiment_2b(W_init, H_init, results_folder, ranks, no_iterations):

    for r in ranks:

        W_init_r = W_init[:, :r]
        H_init_r = H_init[:r, :]

        W, H, divergence_by_it = NMF_divergence(data_matrix, W_init_r, H_init_r, n, m, r, no_iterations, 1,
                                                report_progress=True,
                                                save_progress_to=results_folder)

        save_results(results_folder, W, H, unique_name='r={}_final'.format(r), additional_data_name_to_array_dict={
            'divergence_record' : divergence_by_it}, row_names_list=genes, column_names_list=patients)



if __name__ == "__main__":
    ranks = [50, 100]
    iterations = 5000
    # W_init_a, H_init_a = NNDSVDa_initialization(data_matrix, n, m, max(ranks))
    # experiment_2b(W_init_a, H_init_a, results_folder_a, ranks, iterations)
    W_init_ar, H_init_ar = NNDSVDar_initialization(data_matrix, n, m, max(ranks))
    experiment_2b(W_init_ar, H_init_ar, results_folder_ar, ranks, iterations)

