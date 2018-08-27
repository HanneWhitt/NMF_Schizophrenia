import numpy as np
from matplotlib import pyplot as plt
import os
from shutil import rmtree
import pickle

results_2a = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2a_results/"
results_2b_a = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDa/"
results_2b_ar = "C:/Users/hanne/Documents/PROJECT/Project Data/Experiment_2b_NNDSVDar/"
patients_list_path = 'C:/Users/hanne/Documents/PROJECT/Project Data/CM_patients_list.txt'

results_dict = {'Random': results_2a, 'NNDSVDa': results_2b_a, 'NNDSVDar': results_2b_ar}

main_output_folder = 'C:/Users/hanne/Documents/PROJECT/Figures/Videos/'

ranks = [5, 10, 20, 50, 100]

no_metagenes_to_create_video = 3

iterations = 5000

with open(patients_list_path, 'rb') as fp:
    patients_list = pickle.load(fp)

m_patients = len(patients_list)
patients_to_track = 10

random_state = np.random.RandomState(42)
patient_indexes = random_state.choice(list(range(m_patients)), patients_to_track, replace = False)

print(patient_indexes)

ffmpeg_command = 'ffmpeg -r 50 -i %d.png -c:v libx264 -r 50 -pix_fmt yuv420p {}.mp4'
command_string = ''

for exp_name, exp_path in results_dict.items():

    for r in ranks:

        for metagene_index in list(np.linspace(0, r - 1, no_metagenes_to_create_video).astype(int)):

            name = '{}_init-r={}_MG={}'.format(exp_name, r, metagene_index)

            command_string += ' cd ' + main_output_folder + name + ' & ' + ffmpeg_command.format(name) + ' & '
            print(name + '...')

            plot_title = 'Distribution of Metagene Expression over Patients:\nRank {}, Metagene {}, ' \
                         'Init.: {}'.format(r, metagene_index, exp_name)

            results_path = main_output_folder + name

            if os.path.exists(results_path):
                rmtree(results_path)
            os.mkdir(results_path)

            H_elements_by_it = np.zeros((iterations, m_patients))

            for it in range(iterations):

                H_elements_by_it[it, :] = np.load(exp_path + 'H_r={}_it={}.npy'.format(r, it))[metagene_index,:]

            xmax = np.max(H_elements_by_it) + 10**-20
            bins = np.linspace(0, xmax, 101)
            ymax = 35

            for it in range(iterations):
                plt.clf()
                plt.hist(H_elements_by_it[it, :], label='Iteration {}'.format(it), bins=bins, color='b')
                plt.xlim((0, xmax))
                plt.ylim((0, ymax))
                plt.title(plot_title)
                plt.legend()


                for i in range(patients_to_track):
                    pat_ind = patient_indexes[i]
                    patient_name = patients_list[pat_ind]
                    mg_expression = H_elements_by_it[it, pat_ind]
                    plt.annotate(str(i + 1), xy = (mg_expression, 0), xytext = (mg_expression, ymax - 1 - i), size = 7,
                                 arrowprops=dict(facecolor='black', width = 0.1, headwidth = 0, shrink = 0),
                                 horizontalalignment = 'center')


                plt.savefig(results_path + '/{}.png'.format(it))



command_string = command_string[:-3]
print(command_string)

with open(main_output_folder + 'ffmpeg_command.txt', 'w') as file:
    file.write(command_string)