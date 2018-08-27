import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_main = 'C:/Users/hanne/Documents/PROJECT/Project Data/'
data_2c_CM = data_main + 'Experiment_2c_results/CM/'
figures_path = 'C:/Users/hanne/Documents/PROJECT/Figures/Experiment_2c/'
analysis_data_path = data_main + 'Experiment_2c_analysis/top_genes_significant_metagenes_CM/'


significant_metagenes_2c = [(15, 6), (40, 1)]
sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2500]

for r, metagene_index in significant_metagenes_2c:

    W = pd.read_csv(data_2c_CM + 'W_r={}_final.csv'.format(r), index_col=0)[str(metagene_index)]

    W = W.sort_values(ascending=False)

    n_genes = W.shape[0]

    for sample_size in sample_sizes:
        W_sample = W[:sample_size]
        W_sample.to_csv(analysis_data_path + 'MG{}_exp2c_r{}_top{}.csv'.format(metagene_index, r, sample_size))