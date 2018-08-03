from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np
from NNDSVD_initialization import NNDSVDa_initialization
from nimfa.methods.seeding import nndsvd as nimfa_nndsvd


data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/CM_experiment_2_data.csv"

main_results_folder = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_results/"

data_matrix = pd.read_csv(data_path, index_col=0)

genes = list(data_matrix.index)
patients = list(data_matrix.columns)

n = len(genes)
m = len(patients)
ranks = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]


data_matrix = np.array(data_matrix)


W_init_a, H_init_a = NNDSVDa_initialization(data_matrix, n, m, 150)

nimf = nimfa_nndsvd.Nndsvd()
W_init_a_nimfa, H_init_a_nimfa = nimf.initialize(data_matrix, 150, options={'flag': 1})



W_diffs = W_init_a - W_init_a_nimfa
W_rel_diffs = (W_init_a - W_init_a_nimfa) / W_init_a_nimfa

print('W diff avg{}'.format(np.average(W_diffs)))
print('W diff max{}'.format(np.max(W_diffs)))
print('W RELATIVE diff avg{}'.format(np.average(W_rel_diffs)))
print('W RELATIVE diff max{}'.format(np.max(W_rel_diffs)))


H_diffs = H_init_a - H_init_a_nimfa
H_rel_diffs = (H_init_a - H_init_a_nimfa) / H_init_a_nimfa

print('H diff avg{}'.format(np.average(H_diffs)))
print('H diff max{}'.format(np.max(H_diffs)))
print('H RELATIVE diff avg{}'.format(np.average(H_rel_diffs)))
print('H RELATIVE diff max{}'.format(np.max(H_rel_diffs)))


# for r in ranks:
# 
#     random = np.random.RandomState(42)
# 
#     W_init = random.uniform(0, 1, (n, r))
#     H_init = random.uniform(0, 1, (r, m))
# 
#     W, H, divergence_by_it = NMF_divergence(data_matrix, W_init, H_init, n, m, r, 5000, 1, report_progress=True,
#                                             save_progress_to=main_results_folder)
# 
#     save_results(main_results_folder, W, H, unique_name='r={}_final'.format(r), additional_data_name_to_array_dict={
#         'divergence_record' : divergence_by_it}, row_names_list=genes, column_names_list=patients)