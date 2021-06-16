from __future__ import division, print_function

import os
import sys
import bilby
import deepdish
import numpy as np
import logging
import deepdish
import pandas as pd
import math
import sklearn
import monk
import seaborn as sns

import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'outdir1'
label  = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'

bilby.utils.setup_logger(outdir=outdir, label=label)

ifos = ['CE']

sampling_frequency = 2048.

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
start_time = 1198800017     # for one year beginning
end_time = 1230336017       # for one year long run.

## Total Duration for data taking is
duration = end_time - start_time
print(duration)

## Divide the duration in number of segments i.e. nseg.
#number of time segment
nseg = 10000

# Duration of each segment is
duration_seg = duration / n_seg
print('Duration of one segment (duration_seg) is {}'.format(duration_seg))

## Converting duration of each segment into bits.
duration_seg = 2**(int(duration_seg)-1).bit_length()
print('Duration of one segment (duration_seg) in bits is {}'.format(duration_seg))

## Number of samples in each segment of duration (i.e. duration_seg) are
n_samples = int(sampling_frequency*duration_seg/2)+1
print('Number of Samples in each duration_seg are {}'.format(n_samples))

# Number of truncated time Segment are
n_seg = np.trunc(duration/duration_seg)
print('Number of truncated time segment are {}'.format(n_seg))

## load the injections
injections = deepdish.io.load('injections_test.hdf5')['injections']
#print(injections)

## Total number of injection signals
n_inj = len(injections)
#print('Total Number of Injections  is :', n_inj)

## Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.,
                          minimum_frequency=2.)

## Create the waveform_generator using a LAL BinaryBlackHole source function
## set duration = duration_seg. for duration =4. there will be an error.
waveform_generator = bilby.gw.WaveformGenerator(duration=duration_seg, sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                waveform_arguments=waveform_arguments)

## Subtraction Parameters ##
''' defining a function to read all max_likelihood from all sample datafiles and save them as new DataFrame
    subtraction_parameters used here are the maximum likelihood parameters.'''

def subtracted_parameters():
    ## reading the max likelihood directory
    max_likelihood_direc = os.path.join(current_direc, 'max_likelihood')
    max_likelihood_direc_data = os.listdir('max_likelihood')
    max_likelihood_direc_data_select = [x for x in max_likelihood_direc_data if x.startswith('33')]

    ## create dictionary object for max_likelihood parameters.
    # subtract_param = dict()

    ## for dataframe
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
            # rint(max_likelihood_sample_data_file_open_read)
            # max_likelihood_sample_data_file_read_column_masses = max_likelihood_sample_data_file_open_read.loc[:,'mass_1': 'mass_2']
            # print(max_likelihood_sample_data_file_read_column_masses)

            ## max_likelihood value lies in last row and readin it. we used data.iloc[-1] for last row instead of tail(1). tail(1) will also read only last row of dataframe.
            max_likelihood_in_last_row = max_likelihood_sample_data_file_open_read.iloc[-1]  # tail(1))
            # print(max_likelihood_in_last_row)
            # print(max_likelihood_in_last_row.columns)
            # print(type(max_likelihood_in_last_row))

            max_likelihood_in_last_row_values = max_likelihood_in_last_row.values
            # print(max_likelihood_in_last_row_values)
            # print(type(max_likelihood_in_last_row_values))
            # print(max_likelihood_in_last_row_values[:, 0:1])

            ## convert to dictionary.
            max_likelihood_in_last_row_dict = max_likelihood_in_last_row.to_dict()  # orient='list'
            #print(max_likelihood_in_last_row_dict)
            # print(type(max_likelihood_in_last_row_dict))
            # print(max_likelihood_in_last_row_dict.keys())
            # print(max_likelihood_in_last_row_dict.values())

            ## save the max_likelihood converted to dictionary above as a dictionary
            # subtract_param = max_likelihood_in_last_row_dict

            ## Save as seperate data frame
            subtract_param.append(max_likelihood_in_last_row)

            i += 1
        # subtract_param = max_likelihood_in_last_row_dict  ## for dict
        pd.DataFrame(subtract_param).reset_index(drop=True)

        ## we droped the index column from data rows extracted and assigned new index.

    # return subtract_param
    #return pd.DataFrame(subtract_param)
    return pd.DataFrame(subtract_param).reset_index(drop=True)

## Subtraction Parameters are
subtraction_param = subtracted_parameters()

## Deleting the maximum log_likelihood parameter from subtraction parameters.
del (subtraction_param['log_likelihood'])
# print('New subtraction parameters are :', subtraction_param)

## Total Number of Subtraction/estimated signals are
sub_inj = len(subtraction_param) # number of subtraction parameters.
print('sub_inj',sub_inj)

## Number of parameters in one subtracted/estimated signal are
n_params = len(subtraction_param.iloc[0])
print('n_params',n_params)

## Changing default n_seg and n_inj
n_seg = 100   # time segment
n_inj = 100   # number of Injection signals

for k in np.arange(100):    # np.arange(nseg):

    #print(' Number of time segment is : ', k)

    t0 = start_time + k * duration_seg
    #print('Time t0 is :', t0)
    t1 = t0 + duration_seg
    # print('Time t1 is :',t1)

    IFOs = bilby.gw.detector.InterferometerList(ifos)

    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration_seg
                                                       , start_time=t0)

    #################################
    # injection Parameter and Signal#
    #################################

    injected = False

    for j in np.arange(100):   # np.arange(n_inj)

        #print("Number of Injection signal is :", j)

        injection_parameters = dict(injections.loc[j])
        #print("Injection Parameters are : ", injection_parameters)

        ## check for merger time and injection time segment range.
        geocent_time = injection_parameters['geocent_time']
        #print("Geocent Time is : ",geocent_time)

        if t0 < injection_parameters['geocent_time'] < t1:

            injected = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than injection_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than injection_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, injection_parameters['geocent_time'], t1))

            print("Number of Injection signal is :", j)
            print("Injection Parameters for Injection Signal {} are {}  ".format(j, injection_parameters))

            ## Redshift to luminosity Distance conversion using bilby
            injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
                injection_parameters['redshift'])

            ## First mass needs to be larger than second mass
            if injection_parameters['mass_1'] < injection_parameters['mass_2']:
                tmp = injection_parameters['mass_1']
                injection_parameters['mass_1'] = injection_parameters['mass_2']
                injection_parameters['mass_2'] = tmp

            alpha_1 =injection_parameters

            injected_signal = IFOs.inject_signal(parameters=injection_parameters, injection_polarizations=None,
                               waveform_generator=waveform_generator)

            #print(injected_signal)
            #print(type(injected_signal))

            ## read injected_signal (list of multiple dictionaries.)
            '''
            for item in injected_signal:
                print(item)
                for key in item:
                    print(key)
                    print(item[key])
            '''

            #IFOs.save_data(outdir=outdir,label=label)
            #IFOs.plot_data(outdir='./outdir1', label=label)

        #else:
        #    print("Above Condition for Injection Parameters is Not Satisfied")

    if injected:
        label = 'inj_segment_' + str(k)
        IFOs.save_data(outdir=outdir,label=label)
        IFOs.plot_data(outdir='./outdir1', label=label)


    ##############################
    # Sample Parameter and Signal#
    ##############################

    subtracted = False

    for x in np.arange(sub_inj):   # np.arange(sub_inj)

        # print("Number of subtraction signal is :", x)

        ## convertind subtraction_param DataFrame to dictionary
        subtraction_parameters = dict(subtraction_param.loc[x])
        #print("Subtraction Parameters are : ", subtraction_parameters)

        ## Change the luminosity_distance for reducing the signal from Noise of detector.
        subtraction_parameters['luminosity_distance'] = float(subtraction_parameters['luminosity_distance'])

        ## check for merger time and subtraction time segment range.
        geocent_time = subtraction_parameters['geocent_time']
        #print("Geocent Time is : ",geocent_time)

        if t0 < subtraction_parameters['geocent_time'] < t1:

            subtracted = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than subtraction_parameter_geocent_time which is less than t1')
            #print("t0 = {} is less than subtraction_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, subtraction_parameters['geocent_time'], t1))

            print("Number of Subtraction signal is :", x)
            print("Subtraction parameters for subtraction signal {} are {} ".format(x,subtraction_parameters))

            ## First mass needs to be larger than second mass (just to cross check)
            if subtraction_parameters['mass_1'] < subtraction_parameters["mass_2"]:
                tmp = subtraction_parameters['mass_1']
                subtraction_parameters['mass_1'] = subtraction_parameters['mass_2']
                subtraction_parameters['mass_2'] = tmp

            alpha_2 = subtraction_parameters
            # print(alpha_2)

            subtracted_signal = IFOs.subtract_signal(parameters=subtraction_parameters, injection_polarizations=None,
                                                     waveform_generator=waveform_generator)  # to be derived from bilby's inject_signal
            #print(subtracted_signal)
            #print(type(subtracted_signal))

            #IFOs.save_data(outdir=outdir,label=label1)
            #IFOs.plot_data(outdir='./outdir1', label=label1)

        #else:
        #    print("Above Condition for Subtracted Parameters is Not Satisfied")

    if subtracted:
        label1 = 'sub_segment_' + str(k)
        IFOs.save_data(outdir=outdir,label=label1)
        IFOs.plot_data(outdir='./outdir1', label=label1)

    #############################################
    # Projection Parameter Derivative and Signal#
    #############################################

    projected = False

    for z in np.arange(sub_inj):  # np.arange(sub_inj)

        # print("Number of projected signal is :", z)

        ## convertind subtraction_param DataFrame to dictionary
        projection_parameters = dict(subtraction_param.loc[z])
        # print("Projection Parameters are : ", subtraction_parameters)

        # proj_param = np.array(list(projection_parameters.values()))
        # print(proj_param)
        # projected_parameters = bilby.core.utils.derivatives(vals=proj_param, func=waveform_generator)

        ## Change the luminosity_distance for reducing the signal from Noise of detector.
        projection_parameters['luminosity_distance'] = float(projection_parameters['luminosity_distance'] / 1000)

        ## check for merger time and projection time segment range.
        geocent_time = projection_parameters['geocent_time']
        # print("Geocent Time is : ",geocent_time)

        if t0 < projection_parameters['geocent_time'] < t1:

            projected = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than projection_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than projection_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, subtraction_parameters['geocent_time'], t1))

            print("Number of Projection signal is :", z)
            print("Projection parameters for projection signal {} are {} ".format(z, projection_parameters))

            ## First mass needs to be larger than second mass (just to cross check)
            if projection_parameters['mass_1'] < projection_parameters["mass_2"]:
                tmp = projection_parameters['mass_1']
                projection_parameters['mass_1'] = projection_parameters['mass_2']
                projection_parameters['mass_2'] = tmp

            ## Deleting the 'log_likelihood' parameter from projection_parameters.
            del (projection_parameters['log_likelihood'])
            print('New projection parameters are :', projection_parameters)

            ## make few changes in bilby.core.utils.derivatives
            ## generate a frequency-domain waveform
            # waveform_derivatives = bilby.core.utils.projection_derivatives(vals_dict=projection_parameters, func=waveform_generator.frequency_domain_strain)

            count = 0
            waveform_derivatives = [0 for z1 in range(len(IFOs))]
            for ifo in IFOs:
                def func(parameters):
                    polarizations = waveform_generator.frequency_domain_strain(parameters)

                    return ifo.get_detector_response(polarizations, parameters)

                # print("Projection parameters for func are :", func(projection_parameters))

                waveform_derivatives[count] = bilby.core.utils.projection_derivatives(projection_parameters, func)
                count += 1

                derivatives = np.squeeze(np.array(waveform_derivatives))

                print("Derivatives are :", derivatives)
                print("Desrivative type is ", type(derivatives))
                print("Size of derivatives is :", np.size(derivatives))
                print("Shape of derivatives is :", np.shape(derivatives))
                print('Data type of derivatives is :', derivatives.dtype)

                ## Calculation of Fisher Matrix : A scalar product of signal model w.r.t. model parameters.
                n_params = len(derivatives[:, 1])  # number of parameters
                fisher_matrix = np.zeros((n_params, n_params))

                print('The number of parameters is :', n_params)
                print('Fisher Matrix is :', fisher_matrix)
                print('size of fisher matrix is', np.size(fisher_matrix))
                print('shape of fisher matrix is', np.shape(fisher_matrix))

                ## For every parameter in n_params
                ## iterate through rows
                for q in range(n_params):
                    ## iterate through columns
                    for p in range(q, n_params):
                        # print('i is :', q)
                        # print('p is :', p)

                        PSD = ifo.power_spectral_density

                        prod = bilby.gw.utils.inner_product(derivatives[q], derivatives[p],
                                                            waveform_generator.frequency_array, PSD)
                        fisher_matrices[count, q, p] = prod
                        fisher_matrices[count, p, q] = prod

                count += 1

            fisher_matrix_network = np.zeros((n_params, n_params))
            for k in range(len(IFOs)):
                fisher_matrix_network += fisher_matrices[k, :, :]

            fisher_matrix_norm = np.zeros((n_params, n_params))
            for q in range(n_params):
                for p in range(n_params):
                    fisher_matrix_norm[q, p] = fisher_matrix_network[q, p] / np.sqrt(
                        fisher_matrix_network[p, p] * fisher_matrix_network[q, q])

            print('Fisher Matrix is :', fisher_matrix_norm)
            print(type(fisher_matrix))

            correlation_matrices[z, :, :] = np.linalg.inv(fisher_matrix_norm)

            '''
            # To check if matrix is symmetric.
            symmetric = True
            for m1 in range(len(fisher_matrix)):
                for n1 in range(len(fisher_matrix[m1])):
                    if fisher_matrix[m1][n1] != fisher_matrix[n1][m1]:
                        symmetric = False
                        break
                if not symmetric:
                    break
            print('Is Fisher Matrix  symmetric ? : {}'.format(symmetric))
            '''

            ## Invert the fisher_information_matrix
            fisher_matrix_inverse = np.linalg.inv(fisher_matrix)
            print('fisher_matrix_inverse is : {}'.format(fisher_matrix_inverse))

            ## Transpose of Fisher Matrix
            fisher_information_matrix = np.transpose(fisher_matrix)
            print('Fisher Matrix transpose is {}'.format(fisher_information_matrix))

            # Check if Fisher matrix is equal to its Transpose.
            if fisher_matrix_inverse.all() == fisher_matrix.all():
                print('fisher matrix is an invertible symmetric matrix.')
            else:
                print('fisher matrix is not an invertible symmetric matrix.'
                      'Inverse does not exist because fisher matrix is a singular matrix'
                      '(i.e. determinant of fisher_matrix = 0.')

            # IFOs.save_data(outdir=outdir,label=label2)
            # IFOs.plot_data(outdir='./outdir1', label=label2)

        # else:
        #    print("Above Condition for Projected Parameters is Not Satisfied")

    if projected:
        label1 = 'proj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label2)
        IFOs.plot_data(outdir='./outdir1', label=label2)
