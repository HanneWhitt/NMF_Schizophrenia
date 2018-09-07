from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np
from NNDSVD_initialization import NNDSVDar_initialization
import os
import pickle


data_main = "C:/Users/hanne/Documents/PROJECT/Project Data/"
results_folder_main = data_main + "Experiment_2d_results/"

# CM_rpkm_matrix_path = data_main + "CM_experiment_2_data.csv"
# CM_ensg_to_go_path = data_main + "ENSG_to_go_CM.csv"
# CM_results_folder = results_folder_main + 'CM/'
# CM_genes_with_GO_annotations = list(set(pd.read_csv(CM_ensg_to_go_path, index_col=0)['ENSEMBL']))
# with open(data_main + "CM_genes_with_GO_annotations.txt", "wb") as fp:
#     pickle.dump(CM_genes_with_GO_annotations, fp)

LI_rpkm_matrix_path = data_main + "LI_experiment_2_data.csv"
LI_ensg_to_go_path = data_main + "ENSG_to_go_LI.csv"
LI_results_folder = results_folder_main + 'LI/'
LI_genes_with_GO_annotations = list(set(pd.read_csv(LI_ensg_to_go_path, index_col=0)['ENSEMBL']))
with open(data_main + "LI_genes_with_GO_annotations.txt", "wb") as fp:
    pickle.dump(LI_genes_with_GO_annotations, fp)


def experiment_2d(ranks, no_iterations, rpkm_matrix_path, genes_with_GO_annotations, results_folder, repeat_no):

    rpkm_matrix = pd.read_csv(rpkm_matrix_path, index_col=0).loc[genes_with_GO_annotations]

    patients = list(rpkm_matrix.columns)

    n = len(genes_with_GO_annotations)
    m = len(patients)

    rpkm_matrix = np.array(rpkm_matrix)

    random_state = np.random.RandomState(repeat_no)

    for r in ranks:

        print('\n\nRANK {}, REPEAT {}\n\n'.format(r, repeat_no))

        W_init = random_state.uniform(size = (n, r))
        H_init = random_state.uniform(size = (r, m))

        W, H, divergence_by_it = NMF_divergence(rpkm_matrix, W_init, H_init, n, m, r, no_iterations, 100,
                                                report_progress=True)

        save_results(results_folder, W, H, unique_name='r={}_repeat={}'.format(r, repeat_no),
                     additional_data_name_to_array_dict={'divergence_record' : divergence_by_it},
                     row_names_list=genes_with_GO_annotations,
                     column_names_list=patients)



if __name__ == "__main__":

    iterations = 5000

    # for repeat in [1]:
    #     experiment_2d([100, 150], iterations, CM_rpkm_matrix_path, CM_genes_with_GO_annotations, CM_results_folder,
    #                   repeat)

    ranks = [10, 12, 15, 18, 20, 25, 30, 40, 50, 60]

    # for repeat in range(2, 10):
    #     experiment_2d(ranks, iterations, CM_rpkm_matrix_path, CM_genes_with_GO_annotations, CM_results_folder, repeat)

    for repeat in range(10):
        experiment_2d(ranks, iterations, LI_rpkm_matrix_path, LI_genes_with_GO_annotations, LI_results_folder, repeat)