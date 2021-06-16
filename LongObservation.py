from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import logging
import deepdish
import pandas as pd
import math

import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the name of the simulation.
outdir = './outdir'
label = 'result'
#bilby.utils.setup_logger(outdir=outdir, label=label)

ifos = ['CE']

# Set the duration and sampling frequency of the data segment that we're going to inject the signal into
start_time = 1198800017  ## for one year begninning
end_time = 1230336017  ## for one year long run.

duration = end_time - start_time
print(duration)

time_c = duration/10000
print(time_c)
sampling_frequency = 2048.

if time_c <= time_c + 1:

    ## load the injections
    injections = deepdish.io.load('injections_test.hdf5')['injections']
    time_duration = injections['geocent_time']
    # print(time_duration)

    IFOs = bilby.gw.detector.InterferometerList(ifos)
    #IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=time_c,
    #                                                   start_time=start_time)

    # Specify a cosmological model for z -> d_lum conversion
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Fixed arguments passed into the source model
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50., minimum_frequency=2.)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(duration=time_c, sampling_frequency=sampling_frequency,
                                                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                    waveform_arguments=waveform_arguments)
    #################################
    # injection Parameter and Signal#
    #################################

    for k in np.arange(1,100):
        
        injection_parameters = dict(injections.loc[k])
        #print(injection_parameters)
        #print(type(injection_parameters))
        dist = cosmo.luminosity_distance(injection_parameters['redshift'])
        injection_parameters['luminosity_distance'] = dist.value
        #print(type(injection_parameters['luminosity_distance']))

        # First mass needs to be larger than second mass
        if injection_parameters['mass_1'] < injection_parameters['mass_2']:
            tmp = injection_parameters['mass_1']
            injection_parameters['mass_1'] = injection_parameters['mass_2']
            injection_parameters['mass_2'] = tmp

        start_time_1 = injection_parameters['geocent_time']
        IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=time_c,
                                                           start_time=start_time_1)

        '''
        print(waveform_generator.frequency_domain_strain(parameters=injection_parameters))
        injection_polarizations = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
        print(injection_polarizations)
        print(type(injection_polarizations))
        print(injection_polarizations.keys())

        for key in injection_polarizations:
            injection_polarizations[key] = -1 * injection_polarizations[key]
            print(injection_polarizations[key])
        '''

        injected_signal = IFOs.inject_signal(parameters=injection_parameters, waveform_generator=waveform_generator)
        print(injected_signal)
        #print(type(injected_signal))

        '''
        ## Modifying the values of dictionary within the injected_signal.
        ### we used this for loop because the data from injected_signal line 83 is stored as list of multiple dictionaries.
        for dict_item in injected_signal:
            print(dict_item)
            for key in dict_item:
                print(key)
                print(dict_item[key])
                dict_item[key] *= -1
                print(dict_item[key])
                #key_value = dict_item[key] * -1
                #print("key_value", key_value)
        print(injected_signal)  ##  After changing sign of two polarizations, we stored the latest polarization results again into the injected_signal.
        '''

        #injected_signal_data_file = open('./outdir/CE_frequency_domain_data' + str(k) + '.dat', mode='w')
        ## saving data file to txt format
        #injected_signal_data_file_save = iinjected_signal_data_file.write(str(injected_signal))
        #injected_signal_data_file_save.close()

        #IFOs.save_data(outdir=outdir,label=label)
        #IFOs.save_data(outdir=outdir, label=label)
        #IFOs.plot_data(signal = injected_signal, outdir=outdir, label=label)
        #plot.show()


    ##############################
    # Sample Parameter and Signal#
    # subtraction_parameters used here are the maximum likelihood parameters.
    ##############################

    ## reading the max likelihood directory
    max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
    max_likelihood_direc_data = os.listdir('max_likelihood')
    max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]

    i = 1
    for max_likelihood_data_direc in sorted(max_likelihood_direc_data_select):
        if i <= 100:

            current_max_likelihood_data_direc = os.path.join(max_likelihood_direc, max_likelihood_data_direc)
            print('Current  Data Direc in max_likelihood directory is {}'.format(current_max_likelihood_data_direc))
            max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,'sample_param_' + str(i) + '_result.h5')

            ## Loading the datafile in reading mode.
            max_likelihood_sample_data_file_open = pd.HDFStore(max_likelihood_sample_data_file_name,'r')
            #for max_likelihood_keys in max_likelihood_sample_data_file_open:  ## This command will read main keys and sub keys in the data file
            #    print(max_likelihood_keys)

            max_likelihood_sample_data_file_open_read = pd.read_hdf(max_likelihood_sample_data_file_name, '/data/posterior')
            #rint(max_likelihood_sample_data_file_open_read)
            #max_likelihood_sample_data_file_read_column_masses = max_likelihood_sample_data_file_open_read.loc[:,'mass_1': 'mass_2']
            #print(max_likelihood_sample_data_file_read_column_masses)


            ## max_likelihood value lies in last row and readin it. we used data.iloc[-1] for last row instead of tail(1). tail(1) will also read only last row of dataframe.
            max_likelihood_in_last_row = max_likelihood_sample_data_file_open_read.iloc[-1]    #tail(1))
            #print(max_likelihood_in_last_row)
            #print(max_likelihood_in_last_row.columns)
            #print(type(max_likelihood_in_last_row))

            max_likelihood_in_last_row_values = max_likelihood_in_last_row.values
            #print(max_likelihood_in_last_row_values)
            #print(type(max_likelihood_in_last_row_values))
            #print(max_likelihood_in_last_row_values[:, 0:1])

            ## convert to dictionary.
            max_likelihood_in_last_row_dict = max_likelihood_in_last_row.to_dict() #orient='list'
            #print(max_likelihood_in_last_row_dict)
            #print(type(max_likelihood_in_last_row_dict))
            #print(max_likelihood_in_last_row_dict.keys())
            #print(max_likelihood_in_last_row_dict.values())

            ## subtraction_parameters are
            subtraction_parameters = dict(max_likelihood_in_last_row_dict)
            #subtraction_parameters = max_likelihood_in_last_row_dict        ## for orient='records' or  for orient='list'  '''Check dataframe.to_dict()'''
            print(subtraction_parameters)
            print(type(subtraction_parameters))

            #subtraction_parameters = float(subtraction_parameters[keys])
            #print(subtraction_parameters)

            dist = subtraction_parameters['luminosity_distance']
            #print(dist)
            #print(type(dist))

            ## First mass needs to be larger than second mass (just to cross check)
            if subtraction_parameters['mass_1'] > subtraction_parameters["mass_2"]:
                print("Mass_1 = {} is greater than  Mass_2= {}".format(subtraction_parameters['mass_1'], subtraction_parameters['mass_2']))
            else:
                tmp = subtraction_parameters['mass_1']
                subtraction_parameters['mass_1'] = subtraction_parameters['mass_2']
                subtraction_parameters['mass_2'] = tmp

            subtraction_parameters_start_time = subtraction_parameters['geocent_time']
            #print(subtraction_parameters_start_time)
            IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=time_c,
                                                              start_time=subtraction_parameters_start_time)

            subtracted_signal = IFOs.inject_signal(parameters=subtraction_parameters, waveform_generator=waveform_generator)  # to be derived from bilby's inject_signal
            #print(subtracted_signal)
            #print(type(subtracted_signal))

            ## Modifying the values of dictionary within the injected_signal.
            ### we used this for loop because the data from injected_signal line 83 is stored as list of multiple dictionaries.
            for dict_item in subtracted_signal:
                print(dict_item)
                for key in dict_item:
                    print(key)
                    # print(dict_item[key])
                    # key_value = dict_item[key] * -1
                    # print("key_value", key_value)
                    dict_item[key] *= -1
                    print(dict_item[key])
            print(subtracted_signal)

            # IFOs.save_data(outdir=outdir, label=label)
            # IFOs.plot_data(signal=subtracted_signal, outdir=outdir, label=label)
            # plot.show()

            i += 1
    else:
        print("break")

        

