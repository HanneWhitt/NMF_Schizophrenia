from NMF_divergence import NMF_divergence, save_results
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from analysis_functions_experiment_2 import *

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/CM_experiment_2_data.csv"

main_results_folder = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2a_results/"

data_matrix = pd.read_csv(data_path, index_col=0)

genes = list(data_matrix.index)
patients = list(data_matrix.columns)

n = len(genes)
m = len(patients)

cases, controls = case_control('CM')

no_significant_genes = 0

for gene_index in range(n):
    case_values, control_values = case_control_values(data_matrix, gene_index, cases, controls)

    #mean_case =

    t_statistic, p_value = ttest_ind(case_values, control_values, nan_policy='raise')
    adjusted_p_value = p_value * n

    p_significant = adjusted_p_value <= 0.01

    if p_significant:
        print('Gene {}/{} - SIGNIFICANT, adjusted p: '.format(gene_index, n) + str(adjusted_p_value))

        no_significant_genes += 1





print('\n', no_significant_genes)