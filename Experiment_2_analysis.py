from scipy.stats import ttest_ind
import pandas as pd
from analysis_functions_experiment_2 import *


results_2a = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2a_results/"
results_2b_a = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDa/"
results_2b_ar = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDar/"

exps_dict = {'Random Init.': results_2a, 'NNDSVDa': results_2b_a, 'NNDSVDar': results_2b_ar}

figures_save_loc = "C:/Users/hanne/Documents/PROJECT/Figures/Experiment_2"

divergence_at_5000 = pd.DataFrame()

#Comparing convergence for different initialization methods
for r in [5, 10, 20, 50, 100]:

    label_to_csv_dict = {k:v + 'divergence_record_r={}_final.csv'.format(r) for (k,v) in exps_dict.items()}

    conv_graph_title = 'Convergence from alternative Initialization Methods - Rank ' + str(r)

    conv_data = convergence_data(label_to_csv_dict, show_graph=False, title=conv_graph_title,
                                 it_range = (0, 500), loss_range = (2e6, 4e6))
    print(conv_data)

    divergence_at_5000[str(r)] = conv_data.iloc[4999,:]

print(divergence_at_5000)
input()
cases, controls = case_control('CM')


# A plot of the two components found from rank 2 analysis
# H_mat_df_r2 = pd.read_csv(results_2a + 'H_r={}_final.csv'.format(2), index_col=0)
#
# case_values_0, control_values_0 = case_control_values(H_mat_df_r2, 0, cases, controls)
# case_values_1, control_values_1 = case_control_values(H_mat_df_r2, 1, cases, controls)
#
# plt.scatter(case_values_0, case_values_1, label = 'Schizophrenia')
# plt.scatter(control_values_0, control_values_1, label = 'Control')
# plt.legend()
# plt.show()



# Applying t-tests to results from random initialization (Experiment 2a)

ranks_exp_2a = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150]

t_test_results = t_tests(ranks_exp_2a, cases, controls, results_2a, name = 'Random Initialisation')




# for exp_name, exp_path in exps_dict.items():
#     print('\n\n\n\n' + exp_name)
#     t_tests([5, 10, 20, 50, 100], exp_path, exp_name)




#t_test_results.to_csv(figures_save_loc + '/t_test_results.csv')






