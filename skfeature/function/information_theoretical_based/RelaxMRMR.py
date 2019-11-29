from skfeature.utility.entropy_estimators import *

def relaxmrmr(X, y, **kwargs):
    """
    This function implements the JMI feature selection

    Input
    -----
    X: {numpy array}, shape (num_samples, num_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (num_samples,)
        input class labels
    num_selected_features: {int}
        number of features to select

    Output
    ------
    F: {numpy array}, shape (num_features,)
        index of selected features, F[1] is the most important feature
    """
    
    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 sotres sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    # t4 stores sum_j(I(fj;f|fk)) for each feature f
    t4 = np.zeros(n_features)
    
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    while len(F) < n_selected_features:
        # we assign an extreme small value to j_max to ensure it is smaller than all possible values of feature quality
        j_max = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                if len(F) == 0:
                    # select the feature whose mutual information I(X,Y) is the largest
                    t = t1[i]
                elif len(F) == 1:
                    # select the feature whose mutual information conditioning on one selected features I(Xi,Y|Xj) is the largest
                    t = cmidd(f, y, f_select)
                else:
                    t2[i] += midd(f_select, f)
                    t3[i] += cmidd(f_select, f, y)
                    for fidx in F[:-1]:
                        f_prev = X[:, fidx]
                        t4[i] += cmidd(f, f_select, f_prev)
                        t4[i] += cmidd(f, f_prev, f_select)
                    # calculate j_cmi for feature i (not in F)
                    t = t1[i] - 1.0/len(F)*t2[i] + 1.0/len(F)*t3[i] - 1.0/(len(F)*(len(F)-1))*t4[i]
                if t > j_max:
                    j_max = t
                    idx = i
        F.append(idx)
        f_select = X[:, idx]
    return np.array(F)




