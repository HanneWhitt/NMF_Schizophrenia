import pandas as pd
from final_analysis_and_significance_test import format_confounders_data, complete_significance_test

CM_data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/CM_experiment_2_data.csv"
CM_results_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/DGE_CM/'
CM_pd_path = "C:/Users/hanne/Documents/PROJECT/Project Data/pd_CM.csv"
patient_ref_column_CM = 'DLPFC_RNA_Sequencing_Sample_ID'
confounding_variables_CM = ['Age_of_Death', 'PMI_hrs', 'pH', 'DLPFC_RNA_isolation_RIN']

confounders_data, case_control_1_0_vector = format_confounders_data(CM_pd_path, patient_ref_column_CM,
                                                                  confounding_variables_CM, ['pH'], [], 'Dx', 'SCZ',
                                                                  'Control')

bonferroni_correction_factor = pd.read_csv(CM_data_path, index_col=0).shape[0]
print('Bonferroni correcrion factor:' + str(bonferroni_correction_factor))

complete_significance_test([CM_data_path], 'CM', 'DGE', bonferroni_correction_factor,
                           confounders_data, case_control_1_0_vector, confounding_variables_CM, CM_results_path,
                           t_test_level = 0.01, log_reg_level = 0.05, log_reg_bstrap_tol = 0.025)
