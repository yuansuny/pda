import sys
sys.path.append('fsmethods/')
import numpy as np
import random
import time
import csv
import math
import sys
import os,errno
import scipy.io
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from skfeature.utility.data_discretization import data_discretization
from skfeature.function.information_theoretical_based.RelaxMRMR import relaxmrmr
from skfeature.function.information_theoretical_based.MIM import mim
from skfeature.function.information_theoretical_based.JMI import jmi
from skfeature.function.information_theoretical_based.MRI import mri
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.MIFS import mifs
from skfeature.function.information_theoretical_based.CIFE import cife
from skfeature.function.information_theoretical_based.CMIM import cmim
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.similarity_based.trace_ratio import trace_ratio
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
import VMI_rm
import VMI_gm
import VMI_in
import JMI_rm
import MRMR_rm
import RMRMR_rm


def fileSaving(filename, data_w, writing_model):
    with open(filename, writing_model) as f:
        f_csv=csv.writer(f, quoting=csv.QUOTE_NONE)
        f_csv.writerow(data_w)

if __name__ == '__main__':

    fsMethod = sys.argv[1]
    datName = sys.argv[2]
    print(fsMethod, datName)

    Classifers = ['KNN','SVM']

    filename = 'datasets/%s.mat' % datName
    data = scipy.io.loadmat(filename)

    X = data['data']
    Y = data['labels']
    Y = Y[:,0]
    [numSamples,numFeatures] = X.shape
    numClass = int(max(Y) - min(Y)+1)
    print('Number of Samples: %d' % numSamples)
    print('Number of Features: %d' % numFeatures)
    print('Number of Classes: %d' % len(set(Y)))

    #for continous features;
    #X_dis = data_discretization(X, 5)

    # For discrete features;
    X_dis = X

    filename1 = 'datasets/%s.txt' % datName
    with open(filename1, 'w') as f1:
        for item1 in X_dis:
            for item2 in item1:
                f1.write("%d " % item2)
            f1.write("\n")

    filename2 = 'datasets/%s_labels.txt' % datName
    with open(filename2, 'w') as f2:
        for item in Y:
            f2.write("%d\n" % item)

    #The maximum number of features to be selected
    maxNumSelFeatures = 100

    # Number of runs
    maxRun = 50

    # parameter K for KNN
    n_neighbors = 3


    #feature selection procedure
    if numFeatures < maxNumSelFeatures:
        maxNumSelFeatures = numFeatures

    time0 = time.time()
    if fsMethod == 'VMIrm':
        VMI_rm.VMIrm(nsF=maxNumSelFeatures, filename = datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'VMIgm':
        VMI_gm.VMIgm(nsF=maxNumSelFeatures, filename=datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'VMIin':
        VMI_in.VMIin(nsF=maxNumSelFeatures, filename=datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'RMRMRrm':
        RMRMR_rm.RMRMRrm(nsF=maxNumSelFeatures, filename=datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'JMIrm':
        JMI_rm.JMIrm(nsF=maxNumSelFeatures,  filename=datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'MRMRrm':
        MRMR_rm.MRMRrm(nsF=maxNumSelFeatures,  filename=datName, nS=numSamples, nF=numFeatures, nC=numClass)
    elif fsMethod == 'RelaxMRMR':
        featSelected = relaxmrmr(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'JMI':
        featSelected, J_CMI, MIfy = jmi(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'MRMR':
        featSelected, J_CMI, MIfy = mrmr(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'MIM':
        featSelected, J_CMI, MIfy = mim(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'MRI':
        featSelected = mri(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'MIFS':
        featSelected, J_CMI, MIfy = mifs(X_dis,Y,n_selected_features=maxNumSelFeatures,beta=1)
    elif fsMethod == 'CIFE':
        featSelected, J_CMI, MIfy = cife(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'CMIM':
        featSelected, J_CMI, MIfy = cmim(X_dis,Y,n_selected_features=maxNumSelFeatures)
    elif fsMethod == 'trace_ratio':
        featSelected, feature_score, subset_score = trace_ratio(X_dis,Y,n_selected_features=maxNumSelFeatures)
    else:
        print('The feature selection method %s is not supported' %fsMethod)
        assert(False)
    time1 = time.time()


    filename = "results/sel_features/selFeatures_%s_dataset_%s.csv" %(fsMethod,datName)
    if fsMethod == 'VMIrm' or fsMethod == 'VMIgm' or fsMethod == 'VMIin' \
            or fsMethod == 'JMIrm' or fsMethod == 'MRMRrm' or fsMethod == 'RMRMRrm':
        featSelected = genfromtxt(filename, delimiter=',',dtype=int)
        print(featSelected)
    else:
        fileSaving(filename, featSelected, 'w')
        print('Features selected by %s on dataset %s:' % (fsMethod, datName))
        print(featSelected)


    print('The running time of %s on %s is: %.6f' %(fsMethod,datName,time1-time0))
    filename = "results/running_time/runningTime_%s_dataset_%s.csv" %(fsMethod,datName)
    fileSaving(filename, [time1-time0], 'w')

    for classifer in Classifers:
        if classifer == 'KNN':
            clf = neighbors.KNeighborsClassifier(n_neighbors)
        elif classifer == 'SVM':
            clf = svm.SVC(kernel='linear', C=1)
        else:
            assert(Flase)
        filename = "results/error_rate/%s_error_rate_%s_dateset_%s.csv" %(classifer,fsMethod,datName)
        fileSaving(filename, [], 'w')
        aveScorePerFeat = []
        for i in range(10,maxNumSelFeatures+1):
            selData = X[:,featSelected[:i]]
            aveScoresPerRun = []
            for run in range(maxRun):
                k = random.randint(0,10000)
                cv = ShuffleSplit(n_splits=10, test_size = 0.1, random_state = k)
                scores = cross_val_score(clf, selData, Y, cv=cv)
                scores = 1 - scores
                aveScoresPerRun.append(scores.mean())
            filename = "results/error_rate/%s_error_rate_%s_dateset_%s.csv" %(classifer,fsMethod,datName)
            fileSaving(filename, aveScoresPerRun, 'a')
            aveScorePerFeat.append(np.mean(aveScoresPerRun))
        print('The mean of the %s error rate of %s on %s is: %.8f' %(classifer, fsMethod, datName, np.mean(aveScorePerFeat)))




