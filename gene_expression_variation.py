# This script was used for exploratory analysis of the combined datasets from Common Mind and the Lieber Institute.
# It was also used to make a first attempt at cleaning the data prior to Experiment 1, by rejecting genes which had more
# than 80% zero values and then selecting the top 15000 genes by coefficient of variation in an approach inspired by
# Brunet et al. (2004) and Wang et al. (2012)
# It also produces a dataset where the genes for each patient are randomly permuted, which was used to confirm that
# the decomposition of the real data captured more information than a decomposition of a comparable but random dataset.
# (approach from Frigyesi et al. (2008))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/"
figures_path = "C:/Users/hanne/Documents/PROJECT/Figures/"

X_CM = pd.read_csv(data_path + "CM_matrix.csv", index_col=0)
X_LI = pd.read_csv(data_path + "LI_matrix.csv", index_col=0)

CM_patients = list(X_CM.columns)
LI_patients = list(X_LI.columns)

X = pd.concat([X_CM, X_LI], axis = 1)
del X_LI, X_CM

# Removing genes which register zero expression for all patients
X = X[(X != 0).any(1)]

n_genes = X.shape[0]
m_patients = X.shape[1]

# Calculating means, standard deviations, coefficients of variation for each gene
X_gene_stats = pd.DataFrame()

X_gene_stats['MEAN_GE'] = X.mean(axis = 1)
X_gene_stats['STD_GE'] = X.std(axis = 1)
X_gene_stats['CoV'] = 100*(X_gene_stats['STD_GE']/X_gene_stats['MEAN_GE'])
X_gene_stats['% SPARSITY'] = 100*(X == 0).astype(int).sum(axis=1)/m_patients
X_gene_stats['% SPARSITY CM'] = 100*(X[CM_patients] == 0).astype(int).sum(axis=1)/len(CM_patients)
X_gene_stats['% SPARSITY LI'] = 100*(X[LI_patients] == 0).astype(int).sum(axis=1)/len(LI_patients)




# Histogram plot to show percentage zeros in expression values of genes by frequency
plt.clf()

no_bins = 100
maximum_acceptable_sparsity = 80
y_limit = 3500

bins = np.linspace(0.0, 100.0, no_bins + 1)

N, bins, patches = plt.hist(X_gene_stats['% SPARSITY'], bins = bins)
plt.ylim((0, y_limit))

genes_left = n_genes
for bin_size, bin, patch in zip(N, bins, patches):
    if bin >= maximum_acceptable_sparsity:
        patch.set_facecolor("#FF0000")

plt.title('Frequency of Genes in Dataset by Percentage Sparsity')
plt.xlabel('Percentage Sparsity')
plt.ylabel('Frequency')

no_genes_sparsity_less_than_1pc = int(N[0])
no_genes_with_zero_sparsity = X_gene_stats[X_gene_stats['% SPARSITY'] == 0.0].shape[0]

plt.text(3, 3000, '{} genes with less than 1% zero RPKM values (bar cut)\n(including {} genes with 100% non-zero RPKM values)'.format(no_genes_sparsity_less_than_1pc, no_genes_with_zero_sparsity))

plt.text(30, 700, 'Genes with more than {}% zero RPKM\nvalues excluded from Experiment 1'.format(maximum_acceptable_sparsity))

plt.savefig(figures_path + 'Sparsity Histogram.png', dpi = 500)

plt.show()






# Histogram plot to show Coefficient of variation in expression of genes by frequency

plt.clf()
CoV_min = np.min(X_gene_stats['CoV'])
CoV_max = np.max(X_gene_stats['CoV'])

no_bins = 100
cut_off_for_included_data = 15000

bins_log_scale = np.logspace(np.log10(CoV_min),np.log10(CoV_max), no_bins)

N, bins, patches = plt.hist(X_gene_stats['CoV'], bins = bins_log_scale)
plt.xscale('log')

genes_left = n_genes
for bin_size, bin, patch in zip(N, bins, patches):
    genes_left -= bin_size
    if genes_left <= cut_off_for_included_data:
        patch.set_facecolor("#FF0000")

plt.title('Frequency of Genes in Dataset by\nCoefficient of Variation in Expression Values')
plt.xlabel('Coefficient of Variation')
plt.ylabel('Frequency')

plt.text(400, 600, 'Top {}\ngenes by CoV'.format(cut_off_for_included_data))

plt.savefig(figures_path + 'CoV Histogram.png', dpi = 500)

plt.show()


# Dropping data above chosen sparsity

X_gene_stats = X_gene_stats[X_gene_stats['% SPARSITY'] < maximum_acceptable_sparsity]
X_gene_stats = X_gene_stats.sort_values('% SPARSITY')


# Replotting CoV Histogram

plt.clf()
CoV_min = np.min(X_gene_stats['CoV'])
CoV_max = np.max(X_gene_stats['CoV'])

no_bins = 100
cut_off_for_included_data = 15000

bins_log_scale = np.logspace(np.log10(CoV_min),np.log10(CoV_max), no_bins)

N, bins, patches = plt.hist(X_gene_stats['CoV'], bins = bins_log_scale)
plt.xscale('log')

genes_left = X_gene_stats.shape[0]
for bin_size, bin, patch in zip(N, bins, patches):
    genes_left -= bin_size
    if genes_left <= cut_off_for_included_data:
        patch.set_facecolor("#FF0000")

plt.title('Frequency of Genes in Dataset by\nCoefficient of Variation in Expression Values')
plt.xlabel('Coefficient of Variation')
plt.ylabel('Frequency')

#plt.text(400, 600, 'Top {}\ngenes by CoV'.format(cut_off_for_included_data))

plt.savefig(figures_path + 'CoV Histogram without excluded sparse genes.png', dpi = 500)

plt.show()


# Selecting top 15000 genes from remaining genes by coefficient of variation for first NMF and saving
X_gene_stats = X_gene_stats.sort_values('CoV', ascending=False)
X_gene_stats = X_gene_stats.iloc[:15000, :]
X_experiment_1 = X.loc[X_gene_stats.index]
X_experiment_1.to_csv(data_path + 'Experiment_1_Data.csv')

# A function to randomly permute the columns of the dataset as in Frigyesi et al. 2008
def permute_columns(matrix_dataframe):
    matrix_dataframe = matrix_dataframe.copy()
    matrix_dataframe.apply(np.random.shuffle, axis=0)
    return matrix_dataframe

X_experiment_1_columns_permuted = permute_columns(X_experiment_1)
X_experiment_1_columns_permuted.to_csv(data_path + 'Experiment_1_Data_columns_permuted.csv')






# # # Examining relationship between Coeff. of Var. and % sparsity
#
# for row in range(682):
#     if X_gene_stats['% SPARSITY'][row] > 95:
#         print(X_gene_stats['% SPARSITY'][row])
#         print(X_gene_stats['CoV'][row])
#         #print(np.array(X.iloc[row, :]))
#         input()
#
# plt.clf()
# plt.scatter(X_gene_stats['CoV'], X_gene_stats['% SPARSITY'])
# plt.xscale('log')
# plt.show()





#print(X_gene_stats.head())