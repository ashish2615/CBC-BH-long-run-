from __future__ import print_function
import sys
import os
from pathlib import Path
import numpy as np
import glob
import pandas as pd
import seaborn as sns
from pandas import HDFStore
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


################################
''' Current Working Directory'''
################################

current_direc = os.getcwd()
print('Current Working Directory is : {} '.format(current_direc))
data_direc = os.listdir('.')                  ## This will read current directory in which we are working.(/current_directory)
#print(data_direc)

#########################################
''' Reading the Maximum Likelihood '''
#########################################

## reading the max likelihood directory
max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
print(max_likelihood_direc)

## open the directory to read data directories
max_likelihood_direc_data = os.listdir('max_likelihood')
#print(sorted(max_likelihood_direc_data))
## This will find the files starts with digit 33 in the max_likelihood_direc_data directory.
max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]
#print(len(max_likelihood_direc_data_select))
#print(sorted(max_likelihood_direc_data_select))


#########################################
''' Reading the Maximum Posterior '''
#########################################

## reading the max posterior directory
max_posterior_direc = os.path.join(current_direc, 'max_posterior')
print(max_posterior_direc)

## open the directory to read data directories
max_posterior_direc_data = os.listdir('max_posterior')
#print(sorted(max_posterior_direc_data))
## This will find the files starts with digit 33 in the max_posterior_direc_data directory.
max_posterior_direc_data_select = [x for x in max_posterior_direc_data if x.startswith('33')]
#print(len(max_posterior_direc_data_select))
#print(sorted(max_posterior_direc_data_select))

i =1
for max_likelihood_data_direc, max_posterior_data_direc in zip (sorted(max_likelihood_direc_data_select), sorted(max_posterior_direc_data_select)):

    print(max_likelihood_data_direc)
    print(max_posterior_data_direc)

    ##################################################
    # ''' Reading the Maximum likelihood Data files'''#
    ##################################################

    current_max_likelihood_data_direc = os.path.join(max_likelihood_direc,
                                                     max_likelihood_data_direc)  ## This  command will call the data directories.
    #print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))

    max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc, 'sample_param_' + str(i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
    #print('Maximum Likelihood Sample Data File Name is  {}'.format(max_likelihood_sample_data_file_name))

    ## Loading the datafile in reading mode.
    max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name,'r')  ## This command will load the data file in read mode.
    # print('Maximum Likelihood Sample Data file opened is {}'.format(max_likelihood_sample_data_file_open))
    for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_likelihood_keys)

    max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
    # print('Maximum Likelihood Sample Data file read is {}'.format(max_likelihood_sample_data_file_open_read))
    # print(max_likelihood_sample_data_file_open_read.head()) # head of the DataFrame.
    # print(max_likelihood_sample_data_file_open_read.columns) # columns of Datafrma.
    max_likelihood_sample_data_file_read_column_mass_1 = max_likelihood_sample_data_file_open_read.loc[:,'mass_1': 'mass_1']
    # print(max_likelihood_sample_data_file_read_column_mass_1)
    max_likelihood_sample_data_file_read_column_values = max_likelihood_sample_data_file_read_column_mass_1.values
    # print(max_likelihood_sample_data_file_read_column_values)

    ##########################################################################
    # ''' Reading the Maximum Likelihood SNR Value for the Injection Value'''#
    ##########################################################################

    #max_likelihood_snr_value_data_file_name = os.path.join(current_max_likelihood_data_direc, 'StdOut_' + str(i))
    # print(max_likelihood_snr_value_data_file_name)
    #max_likelihood_snr_value_data_file_read = open(max_likelihood_snr_value_data_file_name, 'r').read()
    # print(max_likelihood_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    #max_likelihood_snr_value_data_file_value = max_likelihood_snr_value_data_file_read[88:]
    # print(max_likelihood_snr_value_data_file_value)

    ##################################################
    # ''' Reading the Maximum Posterior Data files'''#
    ##################################################

    current_max_posterior_data_direc = os.path.join(max_posterior_direc, max_posterior_data_direc)  ## This  command will call the data directories.
    #print('Current  Data Direc in max_posterior directory is {}'.format(current_max_posterior_data_direc))
    max_posterior_sample_data_file_name = os.path.join(current_max_posterior_data_direc, 'sample_param_' + str(i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
    #print('Maximum Posteriore Sample Data File Name is  {}'.format(max_posterior_sample_data_file_name))

    ## Loading the datafile in reading mode.
    max_posterior_sample_data_file_open = pd.HDFStore(max_posterior_sample_data_file_name, 'r')  ## This command will load the data file in read mode.
    # print('Maximum Posterior Sample Data file opened is {}'.format(max_posterior_sample_data_file_open))
    for max_posterior_keys in max_posterior_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_posterior_keys)

    max_posterior_sample_data_file_open_read = pd.read_hdf(max_posterior_sample_data_file_name, '/data/posterior')
    #print('Maximum Posterior Sample Data file read is {}'.format(max_posterior_sample_data_file_open_read))
    #print(max_posterior_sample_data_file_open_read.head()) # head of the DataFrame.
    #print(max_posterior_sample_data_file_open_read.columns) # columns of Datafrma.
    max_posterior_sample_data_file_read_column_mass_1 = max_posterior_sample_data_file_open_read.loc[:, 'mass_1': 'mass_1']
    #print(max_posterior_sample_data_file_read_column_mass_1)
    max_posterior_sample_data_file_read_column_values = max_posterior_sample_data_file_read_column_mass_1.values
    #print(max_posterior_sample_data_file_read_column_values)

    #####################################################################
    # ''' Reading the Maximum Posterior SNR Value for Injection Value'''#
    #####################################################################

    #max_posterior_snr_value_data_file_name = os.path.join(current_max_posterior_data_direc, 'StdOut_' + str(i))
    # print(max_posterior_snr_value_data_file_name)
    #max_posterior_snr_value_data_file_read = open(max_posterior_snr_value_data_file_name, 'r').read()
    # print(max_posterior_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    #max_posterior_snr_value_data_file_value = max_posterior_snr_value_data_file_read[88:]
    # print(max_posterior_snr_value_data_file_value)

    subtracted_param = max_likelihood_sample_data_file_read_column_mass_1.sub(max_posterior_sample_data_file_read_column_mass_1)
    #subtracted_param = np.subtract(max_likelihood_sample_data_file_read_column_values, max_posterior_sample_data_file_read_column_values)
    print(subtracted_param)

    i += 1






### SECOND FORMAT ######
"""
#########################################
''' Reading the Maximum Likelihood '''
#########################################

## reading the max likelihood directory
max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
print(max_likelihood_direc)

## open the directory to read data directories
max_likelihood_direc_data = os.listdir('max_likelihood')
#print(sorted(max_likelihood_direc_data))
## This will find the files starts with digit 33 in the max_likelihood_direc_data directory.
max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]
#print(len(max_likelihood_direc_data_select))
#print(sorted(max_likelihood_direc_data_select))


i = 1 ### max_posterior data files
for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):

    # print(max_likelihood_data_direc)                   ## This will give all data directories with in the current working directories
    current_max_likelihood_data_direc = os.path.join(max_likelihood_direc,
                                                     max_likelihood_data_direc)  ## This  command will call the data directories.
    # print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))

    max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                                        'sample_param_' + str(i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
    print('Maximum Likelihood Sample Data File Name is  {}'.format(max_likelihood_sample_data_file_name))

    ## Loading the datafile in reading mode.
    max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name,
                                                       'r')  ## This command will load the data file in read mode.
    # print('Maximum Likelihood Sample Data file opened is {}'.format(max_likelihood_sample_data_file_open))
    for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_likelihood_keys)

    max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
    # print('Maximum Likelihood Sample Data file read is {}'.format(max_likelihood_sample_data_file_open_read))
    # print(max_likelihood_sample_data_file_open_read.head()) # head of the DataFrame.
    # print(max_likelihood_sample_data_file_open_read.columns) # columns of Datafrma.
    max_likelihood_sample_data_file_read_column_mass_1 = max_likelihood_sample_data_file_open_read.loc[:,
                                                         'mass_1': 'mass_1']
    # print(max_likelihood_sample_data_file_read_column_mass_1)
    max_likelihood_sample_data_file_read_column_values = max_likelihood_sample_data_file_read_column_mass_1.values
    # print(max_likelihood_sample_data_file_read_column_values)

    ##########################################################################
    # ''' Reading the Maximum Likelihood SNR Value for the Injection Value'''#
    ##########################################################################

    max_likelihood_snr_value_data_file_name = os.path.join(current_max_likelihood_data_direc, 'StdOut_' + str(i))
    # print(max_likelihood_snr_value_data_file_name)
    max_likelihood_snr_value_data_file_read = open(max_likelihood_snr_value_data_file_name, 'r').read()
    # print(max_likelihood_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    max_likelihood_snr_value_data_file_value = max_likelihood_snr_value_data_file_read[88:]
    # print(max_likelihood_snr_value_data_file_value)

    i+=1


#########################################
''' Reading the Maximum Posterior '''
#########################################

## reading the max posterior directory
max_posterior_direc = os.path.join(current_direc, 'max_posterior')
print(max_posterior_direc)

## open the directory to read data directories
max_posterior_direc_data = os.listdir('max_posterior')
#print(sorted(max_posterior_direc_data))
## This will find the files starts with digit 33 in the max_posterior_direc_data directory.
max_posterior_direc_data_select = [x for x in max_posterior_direc_data if x.startswith('33')]
#print(len(max_posterior_direc_data_select))
#print(sorted(max_posterior_direc_data_select))


i = 1 ### max_posterior data files
for max_posterior_data_direc in sorted(max_posterior_direc_data_select):

    # print(max_posterior_data_direc)                   ## This will give all data directories with in the current working directories
    current_max_posterior_data_direc = os.path.join(max_posterior_direc,
                                                    max_posterior_data_direc)  ## This  command will call the data directories.
    # print('Current  Data Direc in max_posterior directory is {}'.format(current_max_posteriord_data_direc))
    max_posterior_sample_data_file_name = os.path.join(current_max_posterior_data_direc,
                                                       'sample_param_' + str(
                                                           i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
    print('Maximum Posteriore Sample Data File Name is  {}'.format(max_posterior_sample_data_file_name))

    ## Loading the datafile in reading mode.
    max_posterior_sample_data_file_open = pd.HDFStore(max_posterior_sample_data_file_name,
                                                      'r')  ## This command will load the data file in read mode.
    # print('Maximum Posterior Sample Data file opened is {}'.format(max_posterior_sample_data_file_open))
    for max_posterior_keys in max_posterior_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_posterior_keys)

    max_posterior_sample_data_file_open_read = pd.read_hdf(max_posterior_sample_data_file_name, '/data/posterior')
    # print('Maximum Posterior Sample Data file read is {}'.format(max_posterior_sample_data_file_open_read))
    # print(max_posterior_sample_data_file_open_read.head()) # head of the DataFrame.
    # print(max_posterior_sample_data_file_open_read.columns) # columns of Datafrma.
    max_posterior_sample_data_file_read_column_mass_1 = max_posterior_sample_data_file_open_read.loc[:,
                                                        'mass_1': 'mass_1']
    # print(max_posterior_sample_data_file_read_column_mass_1)
    max_posterior_sample_data_file_read_column_values = max_posterior_sample_data_file_read_column_mass_1.values
    # print(max_posterior_sample_data_file_read_column_values)

    #####################################################################
    # ''' Reading the Maximum Posterior SNR Value for Injection Value'''#
    #####################################################################

    max_posterior_snr_value_data_file_name = os.path.join(current_max_posterior_data_direc, 'StdOut_' + str(i))
    # print(max_posterior_snr_value_data_file_name)
    max_posterior_snr_value_data_file_read = open(max_posterior_snr_value_data_file_name, 'r').read()
    # print(max_posterior_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    max_posterior_snr_value_data_file_value = max_posterior_snr_value_data_file_read[88:]
    # print(max_posterior_snr_value_data_file_value)

    i += 1

"""



### THIRD FORMAT ######
"""
#########################################
''' Reading the Maximum Likelihood '''
#########################################

## reading the max likelihood directory
max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
#print(max_likelihood_direc)

## open the directory to read data directories
max_likelihood_direc_data = os.listdir('max_likelihood')
#print(sorted(max_likelihood_direc_data))
## This will find the files starts with digit 33 in the max_likelihood_direc_data directory.
max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]
#print(len(max_likelihood_direc_data_select))
#print(sorted(max_likelihood_direc_data_select))


def max_likelihood_param(i):

    print(i)

    # i = 1 ### max_likelihood data files
    for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):

        #print(max_likelihood_data_direc)                   ## This will give all data directories with in the current working directories
        current_max_likelihood_data_direc = os.path.join(max_likelihood_direc,
                                               max_likelihood_data_direc)           ## This  command will call the data directories.
        #print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))



        max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                             'sample_param_'+str(i)+'_result.h5')      ## This will call the data file with label 'label_result.h5'.
        print('Maximum Likelihood Sample Data File Name is  {}'.format(max_likelihood_sample_data_file_name))

        ## Loading the datafile in reading mode.
        max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name,'r')  ## This command will load the data file in read mode.
        #print('Maximum Likelihood Sample Data file opened is {}'.format(max_likelihood_sample_data_file_open))
        for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            print(max_likelihood_keys)

        max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
        #print('Maximum Likelihood Sample Data file read is {}'.format(max_likelihood_sample_data_file_open_read))
        #print(max_likelihood_sample_data_file_open_read.head()) # head of the DataFrame.
        #print(max_likelihood_sample_data_file_open_read.columns) # columns of Datafrma.
        max_likelihood_sample_data_file_read_column_mass_1 = max_likelihood_sample_data_file_open_read.loc[:, 'mass_1': 'mass_1']
        #print(max_likelihood_sample_data_file_read_column_mass_1)
        max_likelihood_sample_data_file_read_column_values = max_likelihood_sample_data_file_read_column_mass_1.values
        #print(max_likelihood_sample_data_file_read_column_values)

        ##########################################################################
        # ''' Reading the Maximum Likelihood SNR Value for the Injection Value'''#
        ##########################################################################

        max_likelihood_snr_value_data_file_name = os.path.join(current_max_likelihood_data_direc, 'StdOut_' + str(i))
        #print(max_likelihood_snr_value_data_file_name)
        max_likelihood_snr_value_data_file_read = open(max_likelihood_snr_value_data_file_name, 'r').read()
        #print(max_likelihood_snr_value_data_file_read)
        ## Omiting the all character before SNR value in the file we will use
        max_likelihood_snr_value_data_file_value = max_likelihood_snr_value_data_file_read[88:]
        #print(max_likelihood_snr_value_data_file_value)

        return max_likelihood_sample_data_file_read_column_mass_1
        # return max_likelihood_sample_data_file_read_column_values

        i += 1



#########################################
''' Reading the Maximum Posterior '''
#########################################

## reading the max posterior directory
max_posterior_direc = os.path.join(current_direc, 'max_posterior')
#print(max_posterior_direc)

## open the directory to read data directories
max_posterior_direc_data = os.listdir('max_posterior')
#print(sorted(max_posterior_direc_data))
## This will find the files starts with digit 33 in the max_posterior_direc_data directory.
max_posterior_direc_data_select = [x for x in max_posterior_direc_data if x.startswith('33')]
#print(len(max_posterior_direc_data_select))
#print(sorted(max_posterior_direc_data_select))

def max_posterior_param(i):

    print(i)

    # i = 1 ### max_posterior data files
    for max_posterior_data_direc in sorted(max_posterior_direc_data_select):

        #print(max_posterior_data_direc)                   ## This will give all data directories with in the current working directories
        current_max_posterior_data_direc = os.path.join(max_posterior_direc,
                                               max_posterior_data_direc)           ## This  command will call the data directories.
        #print('Current  Data Direc in max_posterior directory is {}'.format(current_max_posteriord_data_direc))
        max_posterior_sample_data_file_name = os.path.join(current_max_posterior_data_direc,
                                             'sample_param_'+str(i)+'_result.h5')      ## This will call the data file with label 'label_result.h5'.
        print('Sample Data File Name is  {}'.format(max_posterior_sample_data_file_name))

        ## Loading the datafile in reading mode.
        max_posterior_sample_data_file_open = pd.HDFStore(max_posterior_sample_data_file_name,'r')  ## This command will load the data file in read mode.
        #print('Maximum Posterior Sample Data file opened is {}'.format(max_posterior_sample_data_file_open))
        for max_posterior_keys in max_posterior_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            print(max_posterior_keys)

        max_posterior_sample_data_file_open_read = pd.read_hdf(max_posterior_sample_data_file_name, '/data/posterior')
        #print('Maximum Posterior Sample Data file read is {}'.format(max_posterior_sample_data_file_open_read))
        #print(max_posterior_sample_data_file_open_read.head()) # head of the DataFrame.
        #print(max_posterior_sample_data_file_open_read.columns) # columns of Datafrma.
        max_posterior_sample_data_file_read_column_mass_1 = max_posterior_sample_data_file_open_read.loc[:, 'mass_1': 'mass_1']
        #print(max_posterior_sample_data_file_read_column_mass_1)
        max_posterior_sample_data_file_read_column_values = max_posterior_sample_data_file_read_column_mass_1.values
        #print(max_posterior_sample_data_file_read_column_values)

        #####################################################################
        # ''' Reading the Maximum Posterior SNR Value for Injection Value'''#
        #####################################################################

        max_posterior_snr_value_data_file_name = os.path.join(current_max_posterior_data_direc, 'StdOut_' + str(i))
        #print(max_posterior_snr_value_data_file_name)
        max_posterior_snr_value_data_file_read = open(max_posterior_snr_value_data_file_name, 'r').read()
        #print(max_posterior_snr_value_data_file_read)
        ## Omiting the all character before SNR value in the file we will use
        max_posterior_snr_value_data_file_value = max_posterior_snr_value_data_file_read[88:]
        #print(max_posterior_snr_value_data_file_value)

        return max_posterior_sample_data_file_read_column_mass_1
        #return max_posterior_sample_data_file_read_column_values

        i += 1



for i in np.arange(1, 4) :
    print(i)

    #print(max_likelihood_param(i).head())
    #print(max_likelihood_param(i).columns)
    #print(max_posterior_param(i).head())
    #print(max_posterior_param(i).columns)

    subtracted_param = max_likelihood_param(i).sub(max_posterior_param(i))
    print(subtracted_param)

"""


### FOURTH FORMAT ######
"""
#########################################
''' Reading the Maximum Likelihood '''
#########################################

## reading the max likelihood directory
max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
#print(max_likelihood_direc)

## open the directory to read data directories
max_likelihood_direc_data = os.listdir('max_likelihood')
#print(sorted(max_likelihood_direc_data))
## This will find the files starts with digit 33 in the max_likelihood_direc_data directory.
max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]
#print(len(max_likelihood_direc_data_select))
#print(sorted(max_likelihood_direc_data_select))


#########################################
''' Reading the Maximum Posterior '''
#########################################

## reading the max posterior directory
max_posterior_direc = os.path.join(current_direc, 'max_posterior')
#print(max_posterior_direc)

## open the directory to read data directories
max_posterior_direc_data = os.listdir('max_posterior')
#print(sorted(max_posterior_direc_data))
## This will find the files starts with digit 33 in the max_posterior_direc_data directory.
max_posterior_direc_data_select = [x for x in max_posterior_direc_data if x.startswith('33')]
#print(len(max_posterior_direc_data_select))
#print(sorted(max_posterior_direc_data_select))



i = 1 ### max_posterior data files
for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):

    #########################################
    #''' Reading the Maximum Likelihood '''
    #########################################

    current_max_likelihood_data_direc = os.path.join(max_likelihood_direc,max_likelihood_data_direc)  ## This  command will call the data directories.
    print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))

    max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                                        'sample_param_' + str(
                                                            i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
    print('Maximum Likelihood Sample Data File Name is  {}'.format(max_likelihood_sample_data_file_name))

    ## Loading the datafile in reading mode.
    max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name,
                                                       'r')  ## This command will load the data file in read mode.
    # print('Maximum Likelihood Sample Data file opened is {}'.format(max_likelihood_sample_data_file_open))
    for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_likelihood_keys)

    max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
    # print('Maximum Likelihood Sample Data file read is {}'.format(max_likelihood_sample_data_file_open_read))
    # print(max_likelihood_sample_data_file_open_read.head()) # head of the DataFrame.
    # print(max_likelihood_sample_data_file_open_read.columns) # columns of Datafrma.
    max_likelihood_sample_data_file_read_column_mass_1 = max_likelihood_sample_data_file_open_read.loc[:,
                                                         'mass_1': 'mass_1']
    # print(max_likelihood_sample_data_file_read_column_mass_1)
    max_likelihood_sample_data_file_read_column_values = max_likelihood_sample_data_file_read_column_mass_1.values
    # print(max_likelihood_sample_data_file_read_column_values)

    ##########################################################################
    # ''' Reading the Maximum Likelihood SNR Value for the Injection Value'''#
    ######### #################################################################

    max_likelihood_snr_value_data_file_name = os.path.join(current_max_likelihood_data_direc, 'StdOut_' + str(i))
    # print(max_likelihood_snr_value_data_file_name)
    max_likelihood_snr_value_data_file_read = open(max_likelihood_snr_value_data_file_name, 'r').read()
    # print(max_likelihood_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    max_likelihood_snr_value_data_file_value = max_likelihood_snr_value_data_file_read[88:]

    # print(max_likelihood_snr_value_data_file_value)


    for max_posterior_data_direc in sorted(max_posterior_direc_data_select):

        #########################################
        #''' Reading the Maximum Posterior '''
        #########################################

        current_max_posterior_data_direc = os.path.join(max_posterior_direc,
                                                        max_posterior_data_direc)  ## This  command will call the data directories.
        # print('Current  Data Direc in max_posterior directory is {}'.format(current_max_posteriord_data_direc))
        max_posterior_sample_data_file_name = os.path.join(current_max_posterior_data_direc,
                                                           'sample_param_' + str(i) + '_result.h5')  ## This will call the data file with label 'label_result.h5'.
        print('Maximum Posteriore Sample Data File Name is  {}'.format(max_posterior_sample_data_file_name))

        ## Loading the datafile in reading mode.
        max_posterior_sample_data_file_open = pd.HDFStore(max_posterior_sample_data_file_name, 'r')  ## This command will load the data file in read mode.
        # print('Maximum Posterior Sample Data file opened is {}'.format(max_posterior_sample_data_file_open))
        for max_posterior_keys in max_posterior_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            print(max_posterior_keys)

        max_posterior_sample_data_file_open_read = pd.read_hdf(max_posterior_sample_data_file_name, '/data/posterior')
        # print('Maximum Posterior Sample Data file read is {}'.format(max_posterior_sample_data_file_open_read))
        # print(max_posterior_sample_data_file_open_read.head()) # head of the DataFrame.
        # print(max_posterior_sample_data_file_open_read.columns) # columns of Datafrma.
        max_posterior_sample_data_file_read_column_mass_1 = max_posterior_sample_data_file_open_read.loc[:,
                                                            'mass_1': 'mass_1']
        # print(max_posterior_sample_data_file_read_column_mass_1)
        max_posterior_sample_data_file_read_column_values = max_posterior_sample_data_file_read_column_mass_1.values
        # print(max_posterior_sample_data_file_read_column_values)

        #####################################################################
        # ''' Reading the Maximum Posterior SNR Value for Injection Value'''#
        #####################################################################

        max_posterior_snr_value_data_file_name = os.path.join(current_max_posterior_data_direc, 'StdOut_' + str(i))
        # print(max_posterior_snr_value_data_file_name)
        max_posterior_snr_value_data_file_read = open(max_posterior_snr_value_data_file_name, 'r').read()
        # print(max_posterior_snr_value_data_file_read)
        ## Omiting the all character before SNR value in the file we will use
        max_posterior_snr_value_data_file_value = max_posterior_snr_value_data_file_read[88:]
        # print(max_posterior_snr_value_data_file_value)


i += 1

"""