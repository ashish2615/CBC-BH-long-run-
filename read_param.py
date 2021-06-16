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


#########################################
''' Reading the Maximum Likelihood '''
#########################################

current_direc = os.getcwd()
print('Current Working Directory is : {} '.format(current_direc))
data_direc = os.listdir('.')                  ## This will read current directory in which we are working.(/current_directory)
print(data_direc)

"""
### First

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

i = 1 ### max_likelihood data files
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
    #print('Sample Data file opened is {}'.format(sample_data_file_open))
    for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
        print(max_likelihood_keys)

    max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
    #print('max_likelihood_sample Data file read is {}'.format(max_likelihood_sample_data_file_open_read))
    #print(max_likelihood_sample_data_file_open_read.head()) # head of the DataFrame.
    #print(max_likelihood_sample_data_file_open_read.columns) # columns of Datafrma.
    max_likelihood_sample_data_file_read_column_masses = max_likelihood_sample_data_file_open_read.loc[:, 'mass_1': 'mass_2']
    #print(max_likelihood_sample_data_file_read_column_masses)
    max_likelihood_sample_data_file_read_column_values = max_likelihood_sample_data_file_read_column_masses.values
    #print(max_likelihood_sample_data_file_read_column_values)

    #################################################
    # Comparing the mass_1 and mass_2 in data series#
    #################################################
    compare_masses = max_likelihood_sample_data_file_open_read["mass_1"].gt(
        max_likelihood_sample_data_file_open_read[
            "mass_2"]).values  # loc[:, 'mass_1': 'mass_1'].gt max_likelihood_sample_data_file_open_read.loc[:, 'mass_2': 'mass_2']].values
    # compare_masses = max_likelihood_sample_data_file_open_read.loc[max_likelihood_sample_data_file_open_read.mass_1 > max_likelihood_sample_data_file_open_read.mass_2] # .values ## or use .index instead of index
    #print('Compare_masses : ', compare_masses)

    ## reading last row of data frame
    max_likelihood_in_last_row = max_likelihood_sample_data_file_open_read.tail(1)
    #print(max_likelihood_in_last_row)
    max_likelihood_in_last_row_dict = max_likelihood_in_last_row.to_dict(orient='dict')
    print(max_likelihood_in_last_row_dict)
    max_likelihood_in_last_row_values = max_likelihood_in_last_row.values
    #print(max_likelihood_in_last_row_values)
    #print(max_likelihood_in_last_row["luminosity_distance"].values)

    ########################################################
    # ''' Reading the Maximum Likelihood SNR Value for the Injection Value'''#
    ########################################################

    #max_likelihood_snr_value_data_file_name = os.path.join(current_max_likelihood_data_direc, 'StdOut_' + str(i))
    #print(max_likelihood_snr_value_data_file_name)
    #max_likelihood_snr_value_data_file_read = open(max_likelihood_snr_value_data_file_name, 'r').read()
    #print(max_likelihood_snr_value_data_file_read)
    ## Omiting the all character before SNR value in the file we will use
    #max_likelihood_snr_value_data_file_value = max_likelihood_snr_value_data_file_read[88:]
    #print(max_likelihood_snr_value_data_file_value)

    #i += 1
"""

"""
### Second

def subtracted_parameters():

    ## reading the max likelihood directory
    max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
    max_likelihood_direc_data = os.listdir('max_likelihood')
    max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]

    ## create dictionary object
    subtract_param = dict()

    i = 1
    for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):
        if i <= 100:
            current_max_likelihood_data_direc = os.path.join(max_likelihood_direc, max_likelihood_data_direc)
            #print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))
            max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                                                'sample_param_' + str(i) + '_result.h5')

            ## Loading the datafile in reading mode.
            max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name, 'r')
            # for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            #    print(max_likelihood_keys)

            max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name,
                                                                    '/data/posterior')
            # rint(max_likelihood_sample_data_file_open_read)
            # max_likelihood_sample_data_file_read_column_masses = max_likelihood_sample_data_file_open_read.loc[:,'mass_1': 'mass_2']
            # print(max_likelihood_sample_data_file_read_column_masses)

            ## max_likelihood value lies in last row and readin it. we used data.iloc[-1] for last row instead of tail(1). tail(1) will also read only last row of dataframe.
            max_likelihood_in_last_row = max_likelihood_sample_data_file_open_read.iloc[-1]  # tail(1))
            #print(max_likelihood_in_last_row)
            # print(max_likelihood_in_last_row.columns)
            # print(type(max_likelihood_in_last_row))

            max_likelihood_in_last_row_values = max_likelihood_in_last_row.values
            # print(max_likelihood_in_last_row_values)
            # print(type(max_likelihood_in_last_row_values))
            # print(max_likelihood_in_last_row_values[:, 0:1])

            ## convert to dictionary.
            max_likelihood_in_last_row_dict = max_likelihood_in_last_row.to_dict()  # orient='list'
            print(max_likelihood_in_last_row_dict)
            # print(type(max_likelihood_in_last_row_dict))
            # print(max_likelihood_in_last_row_dict.keys())
            # print(max_likelihood_in_last_row_dict.values())

            ## save the max_likelihood converted to dictionary above as a dictionary

            subtract_param = max_likelihood_in_last_row_dict

            i += 1

        subtract_param

    return subtract_param

subtracted_parameters()
#print(max_likelihood_parameters)

"""


### Third

def subtracted_parameters():

    ## reading the max likelihood directory
    max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
    max_likelihood_direc_data = os.listdir('max_likelihood')
    max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]

    ## create dictionary object
    subtract_param = []

    i = 1
    for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):
        if i <= 100:
            current_max_likelihood_data_direc = os.path.join(max_likelihood_direc, max_likelihood_data_direc)
            #print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))
            max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                                                'sample_param_' + str(i) + '_result.h5')

            ## Loading the datafile in reading mode.
            max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name, 'r')
            # for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            #    print(max_likelihood_keys)

            max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name,
                                                                    '/data/posterior')
            print(max_likelihood_sample_data_file_open_read_drop_index)
            # max_likelihood_sample_data_file_read_column_masses = max_likelihood_sample_data_file_open_read.loc[:,'mass_1': 'mass_2']
            # print(max_likelihood_sample_data_file_read_column_masses)

            ## max_likelihood value lies in last row and readin it. we used data.iloc[-1] for last row instead of tail(1). tail(1) will also read only last row of dataframe.
            max_likelihood_in_last_row = max_likelihood_sample_data_file_open_read.iloc[-1]  # tail(1))
            print(max_likelihood_in_last_row)
            # print(max_likelihood_in_last_row.columns)
            # print(type(max_likelihood_in_last_row))

            max_likelihood_in_last_row_values = max_likelihood_in_last_row.values
            # print(max_likelihood_in_last_row_values)
            # print(type(max_likelihood_in_last_row_values))
            # print(max_likelihood_in_last_row_values[:, 0:1])

            ## Save as seperate data frame
            subtract_param.append(max_likelihood_in_last_row)


        i += 1
    pd.DataFrame(subtract_param)


    return pd.DataFrame(subtract_param)

subtraction_parameters = subtracted_parameters()
print(subtraction_parameters)
#print(len(subtraction_parameters))
print(type(subtraction_parameters))

