# Contains functions used to analyse results from experiment 1

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats import ttest_ind

# A function to grab and format data on convergence of an NMF run and optionally plot and save a graph
def convergence_data(label_to_csv_dict, graph_save_location = None, show_graph = False, log_scale = True, title = None):

    data = pd.DataFrame()

    plot = graph_save_location != None or show_graph

    if plot:
        plt.clf()
        if title != None:
            plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Divergence / RPKM')
        if log_scale:
            plt.yscale('log')

    for label, csv in label_to_csv_dict.items():

        data_l = pd.read_csv(csv, index_col=0).iloc[:, :1]
        data_l.columns = [label]

        if data.empty:
            data = data_l
            data.index.name = 'Iteration'
        else:
            data = pd.merge(data, data_l, how='outer', left_index=True, right_index=True)

        if plot:
            plt.plot(data.index, data[label], label = label)

    if plot:
        plt.legend()

    if graph_save_location != None:
        plt.savefig(graph_save_location + title + '.png', dpi=500)

    if show_graph:
        plt.show()

    return data



# A function to load two lists - one of the names of schizophrenia patients, one of the names of the controls
def case_control():
    CM_data = pd.read_csv("C:/Users/hanne/Documents/PROJECT/Project Data/pd_CM.csv")
    CM_names = CM_data['DLPFC_RNA_Sequencing_Sample_ID']
    CM_case_control = CM_data['Dx']
    CM_cases = list(CM_names[CM_case_control == 'SCZ'])
    CM_controls = list(CM_names[CM_case_control == 'Control'])

    LI_data = pd.read_csv("C:/Users/hanne/Documents/PROJECT/Project Data/pd_LI.csv")
    LI_names = LI_data['RNum']
    LI_case_control = LI_data['Dx']
    LI_cases = list(LI_names[LI_case_control == 'Schizo'])
    LI_controls = list(LI_names[LI_case_control == 'Control'])

    cases = CM_cases + LI_cases
    controls = CM_controls + LI_controls

    return cases, controls



# A function to get the values of the expression of a given metagene in a given H matrix for cases and for controls
def case_control_values(H_df, metagene_index, cases, controls, log_transform = False):

    if log_transform:
        min_non_zero_value = H_df.replace(0.0, 1000.0).values.min()
        case_values = np.log(H_df[cases].iloc[metagene_index, :].replace(0.0, min_non_zero_value))
        control_values = np.log(H_df[controls].iloc[metagene_index, :].replace(0.0, min_non_zero_value))
    else:
        case_values = np.array(H_df[cases].iloc[metagene_index, :])
        control_values = np.array(H_df[controls].iloc[metagene_index, :])

    return case_values, control_values


# A function to plot a comparative histogram of the distribution of metagene expression values in cases and in controls
def case_control_comparative_histogram(H_df, metagene_index, cases, controls, title, log_transform = False,
                                       graph_save_location = None, show_graph = True):
    case_values, control_values = case_control_values(H_df, metagene_index, cases, controls, log_transform = log_transform)

    plt.clf()
    plt.hist(case_values, alpha=0.5, label='Schizophrenia', bins = 100)
    plt.hist(control_values, alpha=0.5, label='Control', bins = 100)
    plt.title(title)

    if log_transform:
        plt.xlabel('Log Expression Value')
    else:
        plt.xlabel('Expression Value')

    plt.ylabel('Frequency')
    plt.legend()

    if show_graph:
        plt.show()
    if graph_save_location != None:
        plt.savefig(graph_save_location + title + '.png', dpi = 500)



# A function to get paths to results for a given rank and repeat
def results_path_by_index(main_folder, repeat_no, rank, ranks_list):
    r_index = ranks_list.index(rank)
    folder_index = len(ranks_list) * repeat_no + r_index
    results_folder = os.listdir(main_folder)[folder_index]
    full_path = main_folder + results_folder + '/'
    results_paths = [full_path + x for x in os.listdir(full_path)]
    return results_paths

