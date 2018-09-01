import pandas as pd
from final_analysis_and_significance_test import format_confounders_data, complete_significance_test
import numpy as np
import matplotlib.pyplot as plt

# Finding best of 10 NMF runs by final divergence

def find_best_repeats(data_path, ranks, repeats):
    final_divergence_df = pd.DataFrame(index = repeats, columns = ranks)
    for r in ranks:
        for repeat in range(10):
            divergence_record = np.array(pd.read_csv(data_path +
                                                     'divergence_record_r={}_repeat={}.csv'.format(r, repeat)))
            final_divergence_df.loc[repeat, r] = divergence_record[-1, -1]
    best_repeats = final_divergence_df.astype(float).idxmin()

    H_csv_list = []
    W_csv_list = []
    for r in best_repeats.index:
        best_rep = best_repeats[r]
        file_str = '_r={}_repeat={}.csv'.format(r, best_rep)
        H_csv_list.append(data_path + 'H' + file_str)
        W_csv_list.append(data_path + 'W' + file_str)

    return best_repeats, H_csv_list, W_csv_list



if __name__ == "__main__":

    # Common settings
    ranks = [10, 12, 15, 18, 20, 25, 30, 40, 50, 60]
    repeats = list(range(10))
    bonferroni_correction_factor = sum(ranks)

    # exp_2d_CM
    # data_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/'
    # CM_data_path = data_main + 'Experiment_2d_results/CM/'
    # CM_results_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/exp2d_CM/'
    # CM_pd_path = "C:/Users/hanne/Documents/PROJECT/Project Data/pd_CM.csv"
    # patient_ref_column_CM = 'DLPFC_RNA_Sequencing_Sample_ID'
    # confounding_variables_CM = ['Age_of_Death', 'PMI_hrs', 'pH', 'DLPFC_RNA_isolation_RIN']
    #
    # CM_best_repeats, CM_H_csv_list, CM_W_csv_list = find_best_repeats(CM_data_path, ranks, repeats)
    #
    # confounders_data, case_control_1_0_vector = format_confounders_data(CM_pd_path, patient_ref_column_CM,
    #                                                                   confounding_variables_CM, ['pH'], [], 'Dx', 'SCZ',
    #                                                                   'Control')
    #
    # complete_significance_test(CM_H_csv_list, 'CM', 'exp2d', bonferroni_correction_factor,
    #                            confounders_data, case_control_1_0_vector, confounding_variables_CM, CM_results_path,
    #                            list_of_W_csvs=CM_W_csv_list, t_test_level = 0.01, log_reg_level = 0.05,
    #                            log_reg_bstrap_tol = 0.005)


    # # exp_2d_LI
    confounding_variables_LI = ['Age', 'PMI', 'RIN', 'Race', 'SmokingEither']
    patient_ref_column_LI = 'RNum'
    LI_data_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2d_results/LI/'
    LI_results_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/exp2d_LI/'
    LI_pd_path = "C:/Users/hanne/Documents/PROJECT/Project Data/pd_LI.csv"


    LI_best_repeats, LI_H_csv_list, LI_W_csv_list = find_best_repeats(LI_data_path, ranks, repeats)


    confounders_data, case_control_1_0_vector = format_confounders_data(LI_pd_path, patient_ref_column_LI,
                                                                        confounding_variables_LI, [],
                                                                        ['Race', 'SmokingEither'], 'Dx',
                                                                        'Schizo', 'Control')

    complete_significance_test(LI_H_csv_list, 'LI', 'exp2d', bonferroni_correction_factor,
                               confounders_data, case_control_1_0_vector, confounding_variables_LI, LI_results_path,
                               list_of_W_csvs=LI_W_csv_list, t_test_level=0.01, log_reg_level=0.05,
                               log_reg_bstrap_tol=0.005)