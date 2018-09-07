# Implementing Logistic Regression with Newton-Raphson optimisation
# Reference: Elements of Statistical Learning - Hasti, Tibshirani and Friedman, 2017


import numpy as np
from scipy.stats import chi2
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(X, b):
    activation = np.matmul(X, b)
    return np.array(1/(1 + np.exp(-activation)))


def logistic_regression(X, y, max_iterations = 1000, b_init = None, fisher_info_wald_tests = False,
                        report_progress = False):

    N, p = X.shape

    assert y.shape == (N, 1), 'Data dimensions do not match'

    X = np.concatenate((X, np.ones((N, 1))), axis = 1)

    if b_init is None:
        b = np.zeros((p+1, 1))
    else:
        b = np.reshape(b_init, (p+1, 1))

    for it in range(max_iterations):

        p = sigmoid(X, b)

        W = p * (1 - p)
        W = np.diag(W.flatten())

        inv_neg_hessian = np.linalg.inv(np.matmul(X.T, np.matmul(W, X)))

        del_b = np.matmul(inv_neg_hessian, np.matmul(X.T, y - p))

        b += del_b

        if np.max(np.abs(del_b/b)) < 1e-7:
            if report_progress:
                print('Newton Raphson converged after {} iterations'.format(it + 1))
            break


    coefficients = b.flatten()

    if fisher_info_wald_tests:
        diag_elmts_cov_mat = np.diag(inv_neg_hessian)

        standard_errors = np.sqrt(diag_elmts_cov_mat)
        z_statistics = coefficients/standard_errors

        chi_stats = z_statistics**2
        p_values = 1 - chi2.cdf(chi_stats, 1)

        return coefficients, standard_errors, z_statistics, p_values

    else:
        return coefficients


def format_and_save(estimated_coefficients, standard_errors, z_statistics, p_values,
                                 error_estimate_record, coefficient_names, save_loc = None, save_name = None):
    p_plus_one = estimated_coefficients.size
    shp = (1, p_plus_one)
    res = np.vstack([np.reshape(x, shp) for x in [estimated_coefficients, standard_errors, z_statistics, p_values]])
    len_error_record = error_estimate_record.shape[0]
    res = np.vstack((res, error_estimate_record))
    index = ['Coefficient', 'Bootstrap St. E.', 'Z statistic', 'p-value']
    index += ['BSE it - {}'.format(x*1000) for x in range(1, len_error_record + 1)]
    output = pd.DataFrame(res, columns=coefficient_names, index = index)
    if save_loc != None and save_name != None:
        output.to_csv(save_loc + save_name + '.csv')
    return output


def log_reg_with_bootstrap(X, y, coefficient_names, max_random_samples = 1000000, max_iterations = 1000,
                           tolerance = 0.005, b_init = None, save_loc = None, save_name = None,
                           report_progress_main = True, report_progress_all = False):

    if report_progress_main:
        print('\nEstimating coeffients...')
    estimated_coefficients = logistic_regression(X, y, max_iterations, b_init, report_progress=report_progress_all)
    print(estimated_coefficients)

    N, p = X.shape
    random_state = np.random.RandomState(42)
    bootstrap_coefficients = []
    standard_errors = np.ones((1, p + 1))
    error_estimate_record = standard_errors

    for rep in range(max_random_samples):

        selection = random_state.choice(list(range(N)), size=(N), replace=True)
        X_b = X[selection, :]
        y_b = y[selection, :]

        bootstrap_coefficients.append(logistic_regression(X_b, y_b, report_progress=report_progress_all))

        if (rep + 1) % 1000 == 0:
            new_standard_errors = np.std(np.array(bootstrap_coefficients), axis = 0)
            standard_errors = new_standard_errors
            error_estimate_record = np.vstack((error_estimate_record, standard_errors))

            if report_progress_main:
                print('\nBootstrapping: completed fit to random sample {}'.format(rep + 1))
                print('Estimated standard errors: ', standard_errors)

            if rep >= 9999:

                halfway_point = int(rep / 3000) + 1
                error_estimate_record_last_half = error_estimate_record[halfway_point:, :]
                differences_last_half = error_estimate_record_last_half - standard_errors
                rel_diffs_last_half = differences_last_half/standard_errors
                max_change_ratio_last_half = np.max(np.abs(rel_diffs_last_half))
                percent_change_string = str(100*max_change_ratio_last_half)[:4]
                iteration_interval = rep + 1 - halfway_point*1000
                print('Error estimates stable to ±{}% over last {} iterations'.format(percent_change_string,
                                                                                   iteration_interval))
                if max_change_ratio_last_half < tolerance:
                    break

    standard_errors = np.std(np.array(bootstrap_coefficients), axis = 0)

    z_statistics = estimated_coefficients / standard_errors

    chi_stats = z_statistics**2
    p_values = 1 - chi2.cdf(chi_stats, 1)

    error_estimate_record = error_estimate_record[1:,:]

    results = format_and_save(estimated_coefficients, standard_errors, z_statistics, p_values,
                    error_estimate_record, coefficient_names, save_loc, save_name)

    return results, error_estimate_record


def bootstrap_error_vs_iterations(error_estimate_record, coefficient_names, plot_titles = None, save_loc = None,
                                  tolerance = 0.005, show_graph = False, no_labels = False):

    length = error_estimate_record.shape[0]
    halfway_point = int(length / 3) + 1

    for i in range(len(coefficient_names)):

        plt.clf()
        plt.figure(figsize=(8, 6))
        plot_data = error_estimate_record[:,i]
        plt.plot(plot_data, color = 'b')

        final_error_est = plot_data[-1]
        lower_tolerance = final_error_est*(1 - tolerance)
        higher_tolerance = final_error_est*(1 + tolerance)

        tolerance_line_x_vals = [halfway_point, length]

        plt.plot(tolerance_line_x_vals, [final_error_est, final_error_est], color = 'k', linestyle = '-.',
                 label = 'Final value')
        plt.plot(tolerance_line_x_vals, [lower_tolerance, lower_tolerance], color = 'r', linestyle = '-.',
                 label = '{}% Tolerance'.format(100*tolerance))
        plt.plot(tolerance_line_x_vals, [higher_tolerance, higher_tolerance], color = 'r', linestyle = '-.')
        plt.legend()

        if plot_titles == None:
            plot_title = coefficient_names[i]
        else:
            plot_title = plot_titles[i]

        plt.title(plot_title)
        if not no_labels:
            plt.xlabel('Resamplings/×10^3')
            plt.ylabel('Estimate S.E. of Coeff.')
        if save_loc != None:
            save_path = save_loc + plot_title + '.png'
            plt.savefig(save_path, dpi = 500)
            print('Saved to ', save_path)
        if show_graph:
            plt.show()



# Testing code with randomly generated data
if __name__ == '__main__':

    from time import time
    from sklearn.linear_model import LogisticRegression

    figures_path = 'C:/Users/hanne/Documents/PROJECT/Figures/logistic regression/'

    # Generating random data repoducibly. Two groups with a shift between that allows for some overlap, concatenated
    # together and appropriate 0/1 labelling vector created
    dummy_data_dim = (100, 2)
    shift_between_groups = (1, 2)
    random_state = np.random.RandomState(42)
    group_0 = random_state.standard_normal(dummy_data_dim)
    group_1 = random_state.standard_normal(dummy_data_dim) + shift_between_groups
    X_test = np.concatenate((group_0, group_1), axis = 0)
    y_test = np.concatenate((np.zeros((100, 1)), np.ones((100, 1))), axis=0)

    # Testing project code for basic logistic regression
    start_pc = time()
    b, ses, zs, ps = logistic_regression(X_test, y_test, fisher_info_wald_tests=True)
    pc_time = time() - start_pc

    # Getting gradient and intercept of decision boundary
    grad = -b[0]/b[1]
    intercept = -b[-1]/b[1] # For project code, bias is always last weight in vector

    # Plotting graph to show data and resulting decision boundary
    plt.clf()
    title = 'Logistic Regression - Test on Randomly Generated Data'
    plt.scatter(group_0[:, 0], group_0[:, 1], label = 'Class 0', color = 'g')
    plt.scatter(group_1[:, 0], group_1[:, 1], label = 'Class 1', color = 'r')
    var1_points = np.array([min(X_test[:,0]), max(X_test[:,0])])
    var2_values = var1_points*grad + intercept
    plt.plot(var1_points, var2_values, label = 'p(c = 1) = 0.5', color='b')
    plt.legend()
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title(title)
    #plt.savefig(figures_path + title + '.png', dpi = 500)


    # Testing code against sklearn
    start_sk = time()
    sklearn_model = LogisticRegression(C = 1e100, solver = 'newton-cg', tol = 1e-100)
    sklearn_model = sklearn_model.fit(X_test, y_test)
    sk_time = time() - start_sk

    pc_list = b.flatten()
    sklist = np.array(list(sklearn_model.coef_.flatten()) + (list(sklearn_model.intercept_)))

    print(pc_list - sklist)


    print('Project code coefficients: ', b.flatten())
    print('Project code time: ', pc_time)
    print('Sklearn coefficients: ', sklearn_model.coef_.flatten(), sklearn_model.intercept_)
    print('Sklearn time: ', sk_time)


    # Testing bootstrap error estimation
    coefficient_names = ['Variable 1', 'Variable 2', 'Intercept']

    pc_bs_start = time()
    results, er_rec = log_reg_with_bootstrap(X_test, y_test, coefficient_names)#, save_loc=figures_path,
                                            # save_name = 'Results logistic regression with bootsrapping on random '
                                            #             'data.csv')
    bs_time = time() - pc_bs_start


    # Comparing methods of error estimation based on fisher information from single run and based on bootstrap
    print('\nFISHER INFO ERROR ESTIMATION')
    print('Standard Errors: ',ses)
    print('Z-values: ', zs)
    print('p-values: ', ps)

    print('\nBOOTSTRAP ERROR ESTIMATION')
    print(results)

    # Plotting graphs to show how error estimate of bootstrap varies with number of random samples - important to
    # check that bootstrap error estimate is stable to accuracy required.
    #bootstrap_error_vs_iterations(er_rec, coefficient_names, save_loc=figures_path, show_graph=True)
