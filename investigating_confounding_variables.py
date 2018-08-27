from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd
import numpy as np

data_main = "C:/Users/hanne/Documents/PROJECT/Project Data/"
results_main = "C:/Users/hanne/Documents/PROJECT/Figures/Experiment_2c/"

pd_CM = pd.read_csv(data_main + 'pd_CM.csv', index_col='DLPFC_RNA_Sequencing_Sample_ID')


def impute_nan_values(pd, column_to_impute, scz_col, scz_marker, control_marker):

    print('\nReplacing NaN values in column ' + column_to_impute)
    no_nans = 0

    for group in [scz_marker, control_marker]:

        col_values = pd[pd[scz_col] == group][column_to_impute].astype(float)
        no_nans += np.sum(col_values.isnull())
        mean_col_value_within_group = np.mean(col_values)
        group_nan_index = col_values[col_values.isnull()].index
        pd.loc[group_nan_index, column_to_impute] = mean_col_value_within_group

    print('Replaced {} NaN values out of {} total values in column {}\n'.format(no_nans, pd.shape[0],
          column_to_impute))

    return pd

pd_CM = impute_nan_values(pd_CM, 'pH', 'Dx', 'SCZ', 'Control')



categorical_variables_to_check = ['Gender', 'Ethnicity']

for variable in categorical_variables_to_check:

    print('\n\n', variable.upper())

    cross_tab = pd.crosstab(pd_CM['Dx'], pd_CM[variable])
    chi2, p, dof, ex = chi2_contingency(cross_tab)
    ex = pd.DataFrame(ex, index = cross_tab.index, columns=cross_tab.columns)

    print('\nSample Frequency')
    print(cross_tab)
    print('\nExpected Frequencies if Independent')
    print(ex)

    print('\np-value: ', p)


continuous_variables_to_check = ['Age_of_Death', 'PMI_hrs', 'pH', 'DLPFC_RNA_isolation_RIN']

for variable in continuous_variables_to_check:

    print(variable)

    SCZ_values = pd_CM[pd_CM['Dx'] == 'SCZ'][variable].astype(float)
    Control_values = pd_CM[pd_CM['Dx'] == 'Control'][variable].astype(float)

    no_SCZ_patients = len(SCZ_values)
    no_Control_patients = len(Control_values)
    no_patients = no_Control_patients + no_SCZ_patients

    print('Example SCZ values: ', list(SCZ_values[:10]))
    print('Example Control values: ', list(Control_values[:10]))

    SCZ_mean = np.mean(SCZ_values)
    Control_mean = np.mean(Control_values)
    std = np.std(list(SCZ_values) + list(Control_values))
    std_mean = std/np.sqrt(no_patients)
    print('Mean SCZ: ', SCZ_mean)
    print('Mean Control: ', Control_mean)
    print('St. Dev: ', std)
    print('St. Dev. Mean: ', std_mean)

    t_statistic, p_value = ttest_ind(SCZ_values, Control_values, nan_policy='raise')

    print('T statistic: {}, p-value: {}\n'.format(t_statistic, p_value))



