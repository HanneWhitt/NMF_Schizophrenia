import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
from analysis_functions_experiment_2 import case_control
from logistic_regression_for_control_of_confounders import log_reg_with_bootstrap
from investigating_confounding_variables import impute_nan_values




def format_confounders_data(pd_csv, patient_ref_column, confounding_variable_names, columns_to_impute,
                            categoricals, scz_col, scz_marker, control_marker):

    print('Formatting confounding variables data...')
    confounders_data = pd.read_csv(pd_csv, index_col=patient_ref_column)

    for col in columns_to_impute:
        confounders_data = impute_nan_values(confounders_data, col, scz_col, scz_marker, control_marker)

    for col in categoricals:
        binarized = pd.get_dummies(confounders_data[col])
        confounders_data[col] = binarized[binarized.columns[0]]

    case_control_1_0_vector = pd.get_dummies(confounders_data[scz_col])[scz_marker]
    confounders_data = confounders_data[confounding_variable_names]
    print('Complete')

    return confounders_data, case_control_1_0_vector


def complete_significance_test(list_of_H_csvs, CM_or_LI, experiment_code, bonferroni_correction_factor,
                               confounders_data, case_control_1_0_vector, confounder_names, save_loc,
                               list_of_W_csvs=None, t_test_level = 0.01, log_reg_level = 0.05,
                               log_reg_bstrap_tol = 0.005):

    t_test_p_crit_after_bonferroni = t_test_level/bonferroni_correction_factor
    log_reg_p_crit_after_bonferroni = log_reg_level/bonferroni_correction_factor

    total_tests = 0
    cases, controls = case_control(CM_or_LI)

    significance_test_results = []

    for csv_ind, H_csv in enumerate(list_of_H_csvs):

        H_mat_df = pd.read_csv(H_csv, index_col=0)

        assert list(H_mat_df.columns) == list(confounders_data.index) == list(case_control_1_0_vector.index), \
            'Order of patients does not match'

        r, m_patients = H_mat_df.shape
        total_tests += r

        for ind in range(r):

            if experiment_code == 'DGE':
                name = CM_or_LI + '_' + H_mat_df.index[ind]
            else:
                name = 'MG{}_{}_{}_r{}'.format(ind, experiment_code, CM_or_LI, r)

            print('Testing ' + name)

            values = H_mat_df.iloc[ind, :]
            case_values = np.array(values[cases])
            control_values = np.array(values[controls])
            t_statistic, t_test_p_value = ttest_ind(case_values, control_values, nan_policy='raise')

            if t_test_p_value < t_test_p_crit_after_bonferroni:

                overall_mean = np.mean(values)
                SCZ_mean = np.mean(case_values)
                Control_mean = np.mean(control_values)
                control_std = np.std(values)

                coefficient_names = [name + ' expression'] + confounder_names + ['Intercept']

                expression_values = np.reshape(values, (m_patients, 1))
                full_indep_variables = np.hstack((expression_values, np.array(confounders_data)))
                case_control_1_0 = np.reshape(case_control_1_0_vector, (case_control_1_0_vector.size, 1))

                results, error_estimate_record = log_reg_with_bootstrap(full_indep_variables,
                                                                        case_control_1_0,
                                                                        coefficient_names, tolerance=log_reg_bstrap_tol,
                                                                        save_loc=save_loc,
                                                                        save_name = 'log_reg_' + name)

                def LR_result(result_name):
                    return results.loc[result_name, name + ' expression']

                LR_coeff = LR_result('Coefficient')
                coeff_SE = LR_result('Bootstrap St. E.')
                LR_p_value = LR_result('p-value')

                significance_label = ''
                odds_ratio_increase_pc = ''
                odds_ratio_increase_uncertainty_pc = ''

                if LR_p_value < log_reg_p_crit_after_bonferroni:
                    significance_label = 'SIGNIFICANT'
                    odds_ratio_increase = np.exp(LR_coeff*control_std)
                    odds_ratio_increase_uncertainty_pc = 100*odds_ratio_increase*coeff_SE*control_std
                    odds_ratio_increase_pc = (odds_ratio_increase - 1)*100

                    if list_of_W_csvs != None:
                        sorted_gene_names_loc = save_loc + 'top_genes_significant_metagenes/'
                        if not os.path.exists(sorted_gene_names_loc):
                            os.mkdir(sorted_gene_names_loc)
                        W = pd.read_csv(list_of_W_csvs[csv_ind], index_col=0)[str(ind)]
                        W.to_csv(sorted_gene_names_loc + name + '.csv')

                if experiment_code == 'DGE':
                    significance_test_results.append([name, overall_mean, SCZ_mean, Control_mean, control_std,
                                                      t_test_p_value, LR_coeff, coeff_SE, LR_p_value,
                                                      significance_label, odds_ratio_increase_pc,
                                                      odds_ratio_increase_uncertainty_pc])
                else:
                    significance_test_results.append([name, r, ind, overall_mean, SCZ_mean, Control_mean, control_std,
                                                      t_test_p_value, LR_coeff, coeff_SE, LR_p_value,
                                                      significance_label, odds_ratio_increase_pc,
                                                      odds_ratio_increase_uncertainty_pc])

    assert total_tests == bonferroni_correction_factor, 'Bonferroni correction factor does not match ' \
                                                                   'number of metagenes tested!'

    if experiment_code == 'DGE':
        columns = ['Gene', 'Mean Exp.', 'Mean Exp. SCZ', 'Mean Exp. Control', 'St. Dev. Control', 'T-test p value',
                   'L.R. coeff.', 'L.R. coeff. S.E', 'L.R. p value', 'Significance', 'Odds change, %', 'Uncertainty, %']
    else:
        columns = ['Metagene', 'Rank', 'MG index', 'Mean Exp.', 'Mean Exp. SCZ', 'Mean Exp. Control',
                   'St. Dev. Control', 'T-test p value', 'L.R. coeff.', 'L.R. coeff. S.E', 'L.R. p value',
                   'Significance', 'Odds change, %', 'Uncertainty, %']

    significance_test_results = pd.DataFrame(significance_test_results, columns=columns)

    if experiment_code != 'DGE':
        significance_test_results = significance_test_results.sort_values('Rank')

    significance_test_results.to_csv(save_loc + 'final_LR_results_{}_{}.csv'.format(experiment_code, CM_or_LI))

    return significance_test_results







if __name__ == '__main__':

    # exp_2c_CM
    # data_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/'
    # confounding_variables_CM = ['Age_of_Death', 'PMI_hrs', 'pH', 'DLPFC_RNA_isolation_RIN']
    # patient_ref_column_CM = 'DLPFC_RNA_Sequencing_Sample_ID'
    #
    # csv_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2c_results/CM/'
    # results_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/exp2c_CM/'
    # H_csv_list = [csv_main + file for file in os.listdir(csv_main) if 'H_r=' in file]
    # W_csv_list = [csv_main + file for file in os.listdir(csv_main) if 'W_r=' in file]
    #
    # for csv_list in [H_csv_list, W_csv_list]:
    #     for csv in csv_list:
    #         print(csv)
    #
    # pd_path = "C:/Users/hanne/Documents/PROJECT/Project Data/pd_CM.csv"
    #
    # confounders_data, case_control_1_0_vector = format_confounders_data(pd_path, patient_ref_column_CM,
    #                                                                   confounding_variables_CM, ['pH'], [], 'Dx', 'SCZ',
    #                                                                   'Control')
    #
    # bonferroni_correction_factor = sum([5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150])
    #
    # complete_significance_test(H_csv_list, 'CM', 'exp2c', bonferroni_correction_factor,
    #                            confounders_data, case_control_1_0_vector, confounding_variables_CM, results_path,
    #                            list_of_W_csvs=W_csv_list, t_test_level = 0.01, log_reg_level = 0.05,
    #                            log_reg_bstrap_tol = 0.005)




    # exp_2c_LI
    confounding_variables_LI = ['Age', 'PMI', 'RIN', 'Race', 'SmokingEither']
    patient_ref_column_LI = 'RNum'

    csv_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2c_results/LI/'
    results_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/exp2c_LI/'
    H_csv_list = [csv_main + file for file in os.listdir(csv_main) if 'H_r=' in file]
    W_csv_list = [csv_main + file for file in os.listdir(csv_main) if 'W_r=' in file]

    for csv_list in [H_csv_list, W_csv_list]:
        for csv in csv_list:
            print(csv)

    pd_path = "C:/Users/hanne/Documents/PROJECT/Project Data/pd_LI.csv"

    confounders_data, case_control_1_0_vector = format_confounders_data(pd_path, patient_ref_column_LI,
                                                                        confounding_variables_LI, [],
                                                                        ['Race', 'SmokingEither'], 'Dx',
                                                                        'Schizo','Control')

    print(confounders_data[:10])
    print(case_control_1_0_vector)
    input()

    bonferroni_correction_factor = sum([5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150])

    # complete_significance_test(H_csv_list, 'LI', 'exp2c', bonferroni_correction_factor,
    #                            confounders_data, case_control_1_0_vector, confounding_variables_LI, results_path,
    #                            list_of_W_csvs=W_csv_list, t_test_level=0.01, log_reg_level=0.05,
    #                            log_reg_bstrap_tol=0.005)
