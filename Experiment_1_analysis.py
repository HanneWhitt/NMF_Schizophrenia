import pandas as pd
import numpy as np
from NMF_divergence import D, NMF_divergence, save_results
from analysis_functions_experiment_1 import *
from itertools import chain
import os
from matplotlib import pyplot as plt

data_path = "C:/Users/hanne/Documents/PROJECT/Project Data/"

results_path_main = data_path + 'Experiment_1_results/'
results_path_permuted = data_path + 'Experiment_1_results_permuted/'

figures_path = "C:/Users/hanne/Documents/PROJECT/Figures/Experiment_1/"

repeats = list(range(20))
ranks = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
columns = [[(r, 'Real Data'), (r, 'Permuted')] for r in ranks]
columns = [y for x in columns for y in x]


final_divergences = pd.DataFrame(columns = pd.MultiIndex.from_tuples(columns), index = repeats)
final_divergences.index.name = 'Repeat'
final_divergences.columns.name = 'Rank'


for repeat in repeats:

    for r in ranks:

        paths_main = results_path_by_index(results_path_main, repeat, r, ranks)
        paths_permuted = results_path_by_index(results_path_permuted, repeat, r, ranks)





        #Plotting convergence graph
        div_record_main = paths_main[0]
        div_record_permuted = paths_permuted[0]

        title = 'Convergence on 15k Subset - Rank {} Repeat {}'.format(r, repeat)
        conv_data = convergence_data({'Real Data': div_record_main, 'Permuted': div_record_permuted}, title = title)

        final_divergences[r, 'Real Data'][repeat] = conv_data['Real Data'][100]
        final_divergences[r, 'Permuted'][repeat] = conv_data['Permuted'][100]
#
#
#
#
final_div_real = final_divergences.iloc[:, final_divergences.columns.get_level_values(1)=='Real Data']
final_div_permuted = final_divergences.iloc[:, final_divergences.columns.get_level_values(1)=='Permuted']

# best_repeats_by_rank = final_div_real.idxmin()
# print(best_repeats_by_rank)


# plotting minimum final div values for real and permuteed data by rank
min_final_div_real = final_div_real.min()
min_final_div_permuted = final_div_permuted.min()
difference = np.array(min_final_div_permuted) - np.array(min_final_div_real)

plt.plot(ranks, min_final_div_real, label = 'Real Data')
plt.plot(ranks, min_final_div_permuted, label = 'Permuted')
# plt.plot(ranks, difference, label = 'Diff')
#
plt.title('Final Divergence Value against Rank for Real and Permuted Data')
plt.xlabel('Rank')
plt.ylabel('Final Divergence / RPKM')
plt.yscale('log')
plt.legend()
plt.show()



#final_divergences.to_csv(figures_path + 'final_divergences.csv')