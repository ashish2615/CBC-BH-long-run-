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
import math

import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'outdir2'
label  = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'

bilby.utils.setup_logger(outdir=outdir, label=label)

ifos = ['CE']
#ifos = ['CE','ET']

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
start_time = 1198800017     # for one year beginning
end_time = 1230336017       # for one year long run.

duration = end_time - start_time
print(duration)

## Divide the duration in number of segments i.e. nseg.
nseg = 10000
duration_seg = duration/nseg
print(duration_seg)

sampling_frequency = 2048.

## load the injections
injections = deepdish.io.load('injections_test.hdf5')['injections']
#print(injections)
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

# Specify a cosmological model for z -> d_lum conversion
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


## Subtraction Parameters ##
## defining a function to read all max_likelihood from all sample datafiles and save them as new DataFrame
''' subtraction_parameters used here are the maximum likelihood parameters.'''

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

#subtraction_param_drop_index = subtracted_parameters().reset_index()
#print(subtraction_param_drop_index)
#subtraction_param_drop_index_check = subtracted_parameters().reset_index(drop = True)
#print(subtraction_param_drop_index_check

subtraction_param = subtracted_parameters()
#print(subtraction_param)
sub_inj = len(subtraction_param)
#print('length of subtraction param is : ', sub_inj)
#print(type(subtraction_param))


for k in np.arange(100):    # np.arange(nseg):

    #print(' Number of time segment is : ', k)

    t0 = start_time + k * duration_seg
    #print('Time t0 is :', t0)
    t1 = t0 + duration_seg
    # print('Time t1 is :',t1)

    IFOs = bilby.gw.detector.InterferometerList(ifos)

    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration_seg
                                                       , start_time=t0)
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

            # proj_param = np.array(list(projection_parameters.values()))
            # print(proj_param)

            ## make few changes in bilby.core.utils.derivatives
            ## generate a frequency-domain waveform
            # waveform_derivatives = bilby.core.utils.projection_derivatives(vals_dict=projection_parameters, func=waveform_generator.frequency_domain_strain)

            count = 0
            waveform_derivatives = [0 for z1 in range(len(IFOs))]
            for ifo in IFOs:
                def func(parameters):
                    polarizations = waveform_generator.frequency_domain_strain(parameters)

                    return ifo.get_detector_response(polarizations, parameters)

                #print("Projection parameters for func are :", func(projection_parameters))

                waveform_derivatives[count] = bilby.core.utils.projection_derivatives(projection_parameters, func)
                count += 1

                derivatives = np.squeeze(np.array(waveform_derivatives))

                print("Derivatives are :", derivatives)
                print("Desrivative type is ", type(derivatives))
                print("Size of derivatives is :", np.size(derivatives))
                print("Shape of derivatives is :", np.shape(derivatives))
                print('Data type of derivatives is :',derivatives.dtype)

                '''
                #fisher_information_matrix = np.array([])
                ## For every parameter in n_params
                ## iterate through rows
                for q in range(len(derivatives[1, :])):

                    print('Size of q is :', len(derivatives))

                    ## iterate through columns
                    # for j in range(len(derivatives[1,:])):
                    for j in range(len(derivatives[q])):
                        print('Size of j is :', len(derivatives[q]))

                        print('i is :', q)
                        print('j is :', j)

                        PSD = ifo.power_spectral_density
                        # asd_file = '/Users/ashish2615/Desktop/CBC_BH_long_run/outdir1/CE_projected_signal_psd.dat'
                        # power_spectral_density = PowerSpectralDensity(psd_file=psd_file)
                        # power_spectral_density = PSD.from_amplitude_spectral_density_file(asd_file = asd_file)
                        # print(power_spectral_density)
                        # print(type(power_spectral_density))

                        prod = bilby.gw.utils.projection_inner_product(derivatives[len(derivatives)], derivatives[len(derivatives[q])],
                                                                       waveform_generator.frequency_array, PSD)

                        # (derivatives[:,q], derivatives[:,j], waveform_generator.frequency_array, PSD)

                        # fisher_matrix = np.append(fisher_matrix,prod)

                print("Fisher Matrix Product is :", prod)
                print(type(prod))
                print(prod.dtype)

                # np.savetxt('./outdir2/prod', np.array(prod), delimiter=',')
                '''

                n_params = len(derivatives[:, 1])  # number of parameters
                fisher_matrix = np.zeros((n_params, n_params))

                print('The number of parameters is :', n_params)
                print('Fisher Matrix is :',fisher_matrix)
                print('size of fisher matrix is', np.size(fisher_matrix))
                print('shape of fisher matrix is', np.shape(fisher_matrix))

                ## For every parameter in n_params
                ## iterate through rows
                for q in range(n_params):
                    ## iterate through columns
                    for j in range(q, n_params):
                        #print('i is :', q)
                        #print('j is :', j)

                        PSD = ifo.power_spectral_density

                        prod = bilby.gw.utils.inner_product(derivatives[q], derivatives[j],
                                                            waveform_generator.frequency_array, PSD)
                        fisher_matrix[q, j] = prod
                        fisher_matrix[j, q] = prod

                print('Fisher Matrix is :', fisher_matrix)
                print(type(fisher_matrix))

                '''
                # To check if matrix is symmetric.
                symmetric = True
                for x in range(len(fisher_matrix)):
                    for y in range(len(fisher_matrix[x])):
                        if fisher_matrix[x][y] != fisher_matrix[y][x]:
                            symmetric = False
                            break
                    if not symmetric:
                        break
                print('Is Fisher Matrix  symmetric ? : {}'.format(symmetric))
                '''
                ## Transpose of Fisher Matrix
                fisher_information_matrix = np.transpose(fisher_matrix)
                print(fisher_information_matrix)

                # Check if Fisher matrix is equal to its Transpose.
                if fisher_information_matrix.all() == fisher_matrix.all():
                    print('fisher matrix is a symmetric matrix and non-invertible. '
                          'Inverse does not exist because fisher matrix is a singular matrix'
                          '(i.e. determinant of (fisher_matrix = 0).')

                ## Invert the fisher_information_matrix
                #fisher_matrix_inverse = np.linalg.inv(fisher_matrix)
                #print('fisher_matrix_inverse is : {}'.format(fisher_matrix_inverse))

                ## Projection Operator is
                projection = np.identity(n_params) - fisher_matrix * np.matmul(derivatives, np.conj(np.transpose(derivatives)))
                print('Projection is :', projection)

            # print(projected_signal)
            # print(type(projected_signal))

            # IFOs.save_data(outdir=outdir,label=label2)
            # IFOs.plot_data(outdir='./outdir1', label=label2)

        # else:
        #    print("Above Condition for Projected Parameters is Not Satisfied")

    if projected:
        label1 = 'proj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label2)
        IFOs.plot_data(outdir='./outdir2', label=label2)




