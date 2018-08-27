import pandas as pd
from analysis_functions_experiment_2 import case_control, t_tests
from logistic_regression_for_control_of_confounders import log_reg_with_bootstrap, bootstrap_error_vs_iterations
from investigating_confounding_variables import impute_nan_values
import numpy as np
import matplotlib.pyplot as plt
import os

data_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/'
CM_data_path = data_main + 'Experiment_2c_results/CM/'
figures_path = 'C:/Users/hanne/Documents/PROJECT/Figures/Experiment_2c/'
analysis_data_path = data_main + 'Experiment_2c_analysis/'

# Preliminary t-tests: checking which genes appear to be significant before we account for confounding variables
ranks = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]

cases, controls = case_control('CM')
t_test_results = t_tests(ranks, cases, controls, CM_data_path, name = 'Final Run') #, plot_comparative_hists=True,
                         # save_comparative_hists_to=figures_path + 'comparative histograms/')
#t_test_results.to_csv(figures_path + 'Final_Run_t_test_results.csv')



no_significant_results = t_test_results.shape[0]

pd_CM = pd.read_csv(data_main + 'pd_CM.csv', index_col='DLPFC_RNA_Sequencing_Sample_ID')
pd_CM = impute_nan_values(pd_CM, 'pH', 'Dx', 'SCZ', 'Control')

SCZ_Control_1_0_vector = pd.get_dummies(pd_CM['Dx'])['SCZ']

confounding_variables = ['Age_of_Death', 'PMI_hrs', 'pH', 'DLPFC_RNA_isolation_RIN']
pd_CM = pd_CM[confounding_variables]

m_patients = pd_CM.shape[0]

summary_results = []

for result_index in range(no_significant_results):

    r, metagene_index = t_test_results.loc[result_index][['Rank', 'Metagene Index']].astype(int)


    # expression_values = pd.read_csv(CM_data_path + 'H_r={}_final.csv'.format(r), index_col=0)

    # assert list(expression_values.columns) == list(pd_CM.index) == list(SCZ_Control_1_0_vector.index), \
    #     'Order of patients does not match'

    # expression_values = np.reshape(np.array(expression_values)[metagene_index, :], (m_patients, 1))
    # full_indep_variables = np.hstack((expression_values, np.array(pd_CM)))
    # SCZ_Control = np.reshape(SCZ_Control_1_0_vector, (m_patients, 1))

    metagene_name = 'MG{}_exp2c_r{}'.format(metagene_index, r)
    coefficient_names = [metagene_name + ' expression'] + confounding_variables + ['Intercept']
    save_name = 'log_reg_results_' + metagene_name

    print('\n\nLOADING METAGENE {}\n\n'.format(metagene_name))

    results = pd.read_csv(analysis_data_path + save_name + '.csv', index_col=0)
    error_estimate_record = np.array(results.iloc[4:, :])

    # plot_save_loc = analysis_data_path + 'bs_error_plots/' + metagene_name
    # os.mkdir(plot_save_loc)
    #
    # bootstrap_error_vs_iterations(error_estimate_record, coefficient_names,
    #                               save_loc=plot_save_loc + '/', show_graph=False)

    main_statistics = [results.loc[x, metagene_name + ' expression'] for x in ['Coefficient', 'Bootstrap St. E.',
                                                                               'p-value']]
    adjusted_p_value = main_statistics[-1]*sum(ranks)
    summary_results.append([metagene_name, r, metagene_index] + main_statistics + [adjusted_p_value])

    print('\n\n', results, '\n\n')

summary_results = pd.DataFrame(summary_results, columns=['Metagene Name', 'Rank', 'MG Index', 'Coefficient', 'Error',
                                                         'p-value', 'adjusted p-value'])
summary_results.to_csv(figures_path + 'Experiment_2c_Logistic_Regression_Results_Summary.csv')

#log_reg_with_bootstrap(X, y, coefficient_names, max_random_samples = 1000000, max_iterations = 1000,
                           # tolerance = 0.005, b_init = None, save_loc = None, save_name = None,
                           # report_progress_main = True, report_progress_all = False)

