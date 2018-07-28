'''
This script is used to preprocess the data for the second attempt
at applying NMF to the gene expression data. The following modifications are made to the preprocesssing steps with
the hope of improving results:
    1. The two datasets are prepreprocessed and analysed separately. While there were strgon reasons to suggest that
    the
    two were
    compatible and could be concatenated into a single matrix, problems with the results from experiment 1,
    combined with some further analysis, revealed that there are notable differences between the data from the two
    sources. The clearest way to cope with this problem is to simply analyse the two separately. It is hoped that
    findings from one set will be corroborated by the other.
    2. A log transform is applied to the data prior to NMF. This is done after floor and ceiling values are applied
    to the data.
See write up for full discussion on the above points

2^-10 was chosen as a lower cut off for the RPKM values for the log transform. This updated the following numebrs of
values from each dataset

                    no.     total       proportion
Common Mind         378     12379500    3.0534351145e-05
Lieber Institute    765     11340164    6.74593418578e-05

While a lower threshold could have been set, the proportion of updated non-negative values was negligible,
and it was desirable to limit the size of the required shift into positive log space: the larger the required shift,
the smaller the retained variation between values.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/"
figures_path = "C:/Users/hanne/Documents/PROJECT/Figures/Experiment_2/"

X_CM = pd.read_csv(data_path + "CM_matrix.csv", index_col=0)
X_CM = X_CM[(X_CM != 0).any(1)]

X_LI = pd.read_csv(data_path + "LI_matrix.csv", index_col=0)
X_LI = X_LI[(X_LI != 0).any(1)]

CM_patients = list(X_CM.columns)
LI_patients = list(X_LI.columns)


# Plotting a comparative histogram of ALL the elements in the matrices to help decide on floor (and possibly ceiling)
#  values to apply before log transform

log_transform = True
show_graph = True
title = 'Distribution of all matrix elements by dataset'
graph_save_location = figures_path

def all_elements(X, include_zeros = False):
    elements = np.array(X).flatten()
    if not include_zeros:
        elements = elements[elements != 0]
    return elements

def log_scale_bins(data, no_bins, base = 10):
    data_min = np.min(data)
    data_max = np.max(data)
    bins_log_scale = np.logspace(math.log(data_min, base), math.log(data_max, base), no_bins)
    return bins_log_scale

def elements_below_min(data, min):
    no_below = np.sum(np.where(data < min, 1, 0))
    total = len(data)
    proportion_below = no_below/total
    print(no_below, total, proportion_below)

X_CM_elements = all_elements(X_CM)
X_LI_elements = all_elements(X_LI)

base = 2
no_bins = 100
x_lower_power = -12
x_higher_power = 12
tick_interval = 2

elements_below_min(X_CM_elements, base**x_lower_power)
elements_below_min(X_LI_elements, base**x_lower_power)

plt.clf()

plt.hist(X_CM_elements, alpha=0.5, label='Common Mind dataset', bins = log_scale_bins(X_CM_elements, no_bins, base),
         color='r')
plt.hist(X_LI_elements, alpha=0.5, label='Lieber Institute dataset', bins = log_scale_bins(X_LI_elements, no_bins, base),
         color = 'g')
plt.title(title)


plt.xlabel('Expression Value')
plt.xscale('log', basex = base)
plt.xlim(base**(x_lower_power -1), base**(x_higher_power + 1))

plt.xticks([base**x for x in list(range(x_lower_power, x_higher_power+1, tick_interval))])

plt.ylabel('Frequency')
plt.legend()


plt.show()
plt.savefig(graph_save_location + title + '.png', dpi = 500)






def preprocess(data_matrix, sparsity_cut_off, floor_threshold):

    n_genes = data_matrix.shape[0]
    m_patients = data_matrix.shape[1]

    # Removing genes by sparsity
    gene_stats = pd.DataFrame()
    gene_stats['% SPARSITY'] = 100*(data_matrix == 0).astype(int).sum(axis=1)/m_patients
    data_matrix = data_matrix[gene_stats['% SPARSITY'] < sparsity_cut_off]

    print('Removed genes with more than {}% zero values. {} of {} genes remaining'.format(sparsity_cut_off,
          data_matrix.shape[0], n_genes))

    # applying floor threshold
    data_matrix = data_matrix.where(data_matrix > floor_threshold, floor_threshold)

    print('Zero values replaced with floor: ' + str(floor_threshold))

    # Transforming log base 2
    data_matrix = data_matrix.apply(np.log2)

    # Shifting into positive log space
    shift = -int(np.log2(floor_threshold))
    data_matrix = data_matrix + shift

    return data_matrix


CM_preprocessed = preprocess(X_CM, 30, 2**-10)
LI_preprocessed = preprocess(X_LI, 30, 2**-10)



# Plotting a new histogram to check resulting preprocessed data
title = 'Distribution of transformed data'

base = 2
no_bins = 100

X_CM_elements_preproc = all_elements(CM_preprocessed, True)
X_LI_elements_preproc = all_elements(LI_preprocessed, True)


plt.clf()

plt.hist(X_CM_elements_preproc, alpha=0.5, label='Common Mind dataset', bins = no_bins,
         color='r')
plt.hist(X_LI_elements_preproc, alpha=0.5, label='Lieber Institute dataset', bins = no_bins, color = 'g')
plt.title(title)


plt.xlabel('Shifted Log Expression Value')

plt.xlim(xmax=30)

plt.ylabel('Frequency')
plt.legend()


plt.savefig(graph_save_location + title + '.png', dpi = 500)











#
# # Calculating means, standard deviations, coefficients of variation for each gene
#
# X_gene_stats['% SPARSITY CM'] = 100*(X[CM_patients] == 0).astype(int).sum(axis=1)/len(CM_patients)
# X_gene_stats['% SPARSITY LI'] = 100*(X[LI_patients] == 0).astype(int).sum(axis=1)/len(LI_patients)
#
#
#
#
# # Histogram plot to show percentage zeros in expression values of genes by frequency
# plt.clf()
#
# no_bins = 100
# maximum_acceptable_sparsity = 80
# y_limit = 3500
#
# bins = np.linspace(0.0, 100.0, no_bins + 1)
#
# N, bins, patches = plt.hist(X_gene_stats['% SPARSITY'], bins = bins)
# plt.ylim((0, y_limit))
#
# genes_left = n_genes
# for bin_size, bin, patch in zip(N, bins, patches):
#     if bin >= maximum_acceptable_sparsity:
#         patch.set_facecolor("#FF0000")
#
# plt.title('Frequency of Genes in Dataset by Percentage Sparsity')
# plt.xlabel('Percentage Sparsity')
# plt.ylabel('Frequency')
#
# no_genes_sparsity_less_than_1pc = int(N[0])
# no_genes_with_zero_sparsity = X_gene_stats[X_gene_stats['% SPARSITY'] == 0.0].shape[0]
#
# plt.text(3, 3000, '{} genes with less than 1% zero RPKM values (bar cut)\n(including {} genes with 100% non-zero RPKM values)'.format(no_genes_sparsity_less_than_1pc, no_genes_with_zero_sparsity))
#
# plt.text(30, 700, 'Genes with more than {}% zero RPKM\nvalues excluded from Experiment 1'.format(maximum_acceptable_sparsity))
#
# plt.savefig(figures_path + 'Sparsity Histogram.png', dpi = 500)
#
# plt.show()
#
#
#
#
#
#
# # Histogram plot to show Coefficient of variation in expression of genes by frequency
#
# plt.clf()
# CoV_min = np.min(X_gene_stats['CoV'])
# CoV_max = np.max(X_gene_stats['CoV'])
#
# no_bins = 100
# cut_off_for_included_data = 15000
#
# bins_log_scale = np.logspace(np.log10(CoV_min),np.log10(CoV_max), no_bins)
#
# N, bins, patches = plt.hist(X_gene_stats['CoV'], bins = bins_log_scale)
# plt.xscale('log')
#
# genes_left = n_genes
# for bin_size, bin, patch in zip(N, bins, patches):
#     genes_left -= bin_size
#     if genes_left <= cut_off_for_included_data:
#         patch.set_facecolor("#FF0000")
#
# plt.title('Frequency of Genes in Dataset by\nCoefficient of Variation in Expression Values')
# plt.xlabel('Coefficient of Variation')
# plt.ylabel('Frequency')
#
# plt.text(400, 600, 'Top {}\ngenes by CoV'.format(cut_off_for_included_data))
#
# plt.savefig(figures_path + 'CoV Histogram.png', dpi = 500)
#
# plt.show()
#
#
# # Dropping data above chosen sparsity
#
# X_gene_stats = X_gene_stats[X_gene_stats['% SPARSITY'] < maximum_acceptable_sparsity]
# X_gene_stats = X_gene_stats.sort_values('% SPARSITY')
#
#
# # Replotting CoV Histogram
#
# plt.clf()
# CoV_min = np.min(X_gene_stats['CoV'])
# CoV_max = np.max(X_gene_stats['CoV'])
#
# no_bins = 100
# cut_off_for_included_data = 15000
#
# bins_log_scale = np.logspace(np.log10(CoV_min),np.log10(CoV_max), no_bins)
#
# N, bins, patches = plt.hist(X_gene_stats['CoV'], bins = bins_log_scale)
# plt.xscale('log')
#
# genes_left = X_gene_stats.shape[0]
# for bin_size, bin, patch in zip(N, bins, patches):
#     genes_left -= bin_size
#     if genes_left <= cut_off_for_included_data:
#         patch.set_facecolor("#FF0000")
#
# plt.title('Frequency of Genes in Dataset by\nCoefficient of Variation in Expression Values')
# plt.xlabel('Coefficient of Variation')
# plt.ylabel('Frequency')
#
# #plt.text(400, 600, 'Top {}\ngenes by CoV'.format(cut_off_for_included_data))
#
# plt.savefig(figures_path + 'CoV Histogram without excluded sparse genes.png', dpi = 500)
#
# plt.show()
#
#
# # Selecting top 15000 genes from remaining genes by coefficient of variation for first NMF and saving
# X_gene_stats = X_gene_stats.sort_values('CoV', ascending=False)
# X_gene_stats = X_gene_stats.iloc[:15000, :]
# X_experiment_1 = X.loc[X_gene_stats.index]
# X_experiment_1.to_csv(data_path + 'Experiment_1_Data.csv')
#
# # A function to randomly permute the columns of the dataset as in Frigyesi et al. 2008
# def permute_columns(matrix_dataframe):
#     matrix_dataframe = matrix_dataframe.copy()
#     matrix_dataframe.apply(np.random.shuffle, axis=0)
#     return matrix_dataframe
#
# X_experiment_1_columns_permuted = permute_columns(X_experiment_1)
# X_experiment_1_columns_permuted.to_csv(data_path + 'Experiment_1_Data_columns_permuted.csv')
#
#
#
#
#
#
# # # # Examining relationship between Coeff. of Var. and % sparsity
# #
# # for row in range(682):
# #     if X_gene_stats['% SPARSITY'][row] > 95:
# #         print(X_gene_stats['% SPARSITY'][row])
# #         print(X_gene_stats['CoV'][row])
# #         #print(np.array(X.iloc[row, :]))
# #         input()
# #
# # plt.clf()
# # plt.scatter(X_gene_stats['CoV'], X_gene_stats['% SPARSITY'])
# # plt.xscale('log')
# # plt.show()
#
#
#
#

#print(X_gene_stats.head())