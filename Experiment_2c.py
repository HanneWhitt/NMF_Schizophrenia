from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np
from NNDSVD_initialization import NNDSVDar_initialization


data_main = "C:/Users/hanne/Documents/PROJECT/Project Data/"
results_folder_main = data_main + "Experiment_2c_results/"

CM_rpkm_matrix_path = data_main + "CM_experiment_2_data.csv"
CM_ensg_to_go_path = data_main + "ENSG_to_go_CM.csv"
CM_results_folder = results_folder_main + 'CM/'

LI_rpkm_matrix_path = data_main + "LI_experiment_2_data.csv"
LI_ensg_to_go_path = data_main + "ENSG_to_go_LI.csv"
LI_results_folder = results_folder_main + 'LI/'


def experiment_2c(ranks, no_iterations, rpkm_matrix_path, ensg_to_go_path, results_folder):

    genes_with_GO_annotations = list(set(pd.read_csv(ensg_to_go_path, index_col=0)['ENSEMBL']))

    rpkm_matrix = pd.read_csv(rpkm_matrix_path, index_col=0).loc[genes_with_GO_annotations]

    patients = list(rpkm_matrix.columns)

    n = len(genes_with_GO_annotations)
    m = len(patients)

    rpkm_matrix = np.array(rpkm_matrix)

    W_init, H_init = NNDSVDar_initialization(rpkm_matrix, n, m, max(ranks))

    for r in ranks:

        W_init_r = W_init[:, :r]
        H_init_r = H_init[:r, :]

        W, H, divergence_by_it = NMF_divergence(rpkm_matrix, W_init_r, H_init_r, n, m, r, no_iterations, 100,
                                                report_progress=True)

        save_results(results_folder, W, H, unique_name='r={}_final'.format(r), additional_data_name_to_array_dict={
            'divergence_record' : divergence_by_it}, row_names_list=genes_with_GO_annotations,
                     column_names_list=patients)



if __name__ == "__main__":

    iterations = 5000
    ranks = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]

    experiment_2c(ranks, iterations, CM_rpkm_matrix_path, CM_ensg_to_go_path, CM_results_folder)
    experiment_2c(ranks, iterations, LI_rpkm_matrix_path, LI_ensg_to_go_path, LI_results_folder)