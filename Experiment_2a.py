from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np


data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/CM_experiment_2_data.csv"

main_results_folder = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2a_results/"

data_matrix = pd.read_csv(data_path, index_col=0)

genes = list(data_matrix.index)
patients = list(data_matrix.columns)

n = len(genes)
m = len(patients)
ranks = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]


data_matrix = np.array(data_matrix)


for r in ranks:

    random = np.random.RandomState(42)

    W_init = random.uniform(0, 1, (n, r))
    H_init = random.uniform(0, 1, (r, m))

    W, H, divergence_by_it = NMF_divergence(data_matrix, W_init, H_init, n, m, r, 5000, 1, report_progress=True,
                                            save_progress_to=main_results_folder)

    save_results(main_results_folder, W, H, unique_name='r={}_final'.format(r), additional_data_name_to_array_dict={
        'divergence_record' : divergence_by_it}, row_names_list=genes, column_names_list=patients)