from pathlib import Path # For making paths compatible on Windows and Macs
import pandas as pd # For working with DataFrames
import numpy as np # For ease of array manipulation, stats, and some feature extraction
import matplotlib.pyplot as plt # For plotting pretty plots :) 
import scipy.signal as signal # For calculating PSDs and plotting spectrograms

import pyeeg # For pyeeg implemented features
import pickle # For loading and creating pickle files
from neurodsp.spectral import compute_spectrum # For smoothed PSD computation
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time # For neurodsp features


eeg_fs = 250 # Data was recorded at 250 Hz
eeg_chans = ["C3", "Cz", "C4"] # 10-20 system
eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
all_chans = eeg_chans + eog_chans
event_types = {0:"left", 1:"right"}


# Multiple bar graph plotting
def plotMultipleBarGraphs(bars, bar_width, bar_names, group_names, error_values=None, title=None, xlabel=None,
                          ylabel=None):
    if len(bar_names) != len(bars):
        print("group names must be same length as bars")
        return
        # Set position of bar on X axis
    positions = list()
    positions.append(np.arange(len(bars[0])))
    for i, bar in enumerate(bars):
        if i > 0:
            positions.append([x + bar_width for x in positions[i - 1]])

    # Make the plot
    for i, pos in enumerate(positions):
        plt.bar(pos, bars[i], width=bar_width, label=bar_names[i])

    if error_values is not None:
        for i, pos in enumerate(positions):
            plt.errorbar(pos, bars[i], yerr=error_values[i], fmt='.k')

    # Add xticks on the middle of the group bars
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.xticks([r + bar_width for r in range(len(bars[0]))], group_names)

    # Create legend & Show graphic
    plt.legend()
    plt.show()


# Power Bin extraction
def getPowerRatio(eeg_data, binning, eeg_fs=250):
    power, power_ratio = pyeeg.bin_power(eeg_data, binning, eeg_fs)
    return np.array(power_ratio)


def getIntervals(binning):
    intervals = list()
    for i, val in enumerate(binning[:-1]):
        intervals.append((val, binning[i + 1]))
    return intervals


def getPowerBin(eeg_epoch_full_df):
    # Create a dataframe with the event_type and the power bin information for each trial
    power_ratios = {'y': []}
    binning = [0.5, 4, 7, 12, 30]
    intervals = getIntervals(binning)
    for i in range(0, len(eeg_epoch_full_df)):
        event_type = eeg_epoch_full_df['event_type'][i]
        for ch in eeg_chans:
            ratios = getPowerRatio(eeg_epoch_full_df[ch][i][:], binning)
            for j, interval in enumerate(intervals):
                key = ch + "_" + str(interval)
                if key not in power_ratios:
                    power_ratios[key] = list()
                power_ratios[key].append(ratios[j])
        power_ratios['y'].append(eeg_epoch_full_df['event_type'][i])

    power_ratios_df = pd.DataFrame(power_ratios)
    power_ratios_df.head(2)

    # Calculate the standard error means between epochs for each channel from the power ratios obtained previously
    chan_frequency_sems = {}
    chan_frequency_avgs = {}

    for event_type in event_types:
        for ch in eeg_chans:
            for interval in intervals:
                key = ch + "_" + str(interval)
                if key not in chan_frequency_sems:
                    chan_frequency_sems[key] = list()
                    chan_frequency_avgs[key] = list()
                this_data = power_ratios_df[power_ratios_df['y'] == event_type][key]
                sem = np.std(this_data) / np.sqrt(len(this_data))  # Standard Error of Mean calculation
                chan_frequency_sems[key].append(sem)
                chan_frequency_avgs[key].append(np.mean(this_data))

    std_err_df = pd.DataFrame(chan_frequency_sems)
    avg_df = pd.DataFrame(chan_frequency_avgs)

    # Plot average power ratios for each electrode
    for chan in eeg_chans:
        chan_of_interest = chan
        event_power_ratios = {}
        event_sems = {}
        power_ratios_for_chan = []
        sem_for_chan = []
        for event_type in event_types:
            if event_type not in event_power_ratios:
                event_power_ratios[event_type] = []
                event_sems[event_type] = []
            for interval in intervals:
                key = chan_of_interest + "_" + str(interval)
                event_power_ratios[event_type].append(avg_df[key][event_type])
                event_sems[event_type].append(std_err_df[key][event_type])

        event_sems_df = pd.DataFrame(event_sems)
        event_power_ratios_df = pd.DataFrame(event_power_ratios)

        plt.title(chan_of_interest)
        plt.ylim((0, 0.5))
        plt.ylabel("Power Ratio")
        plotMultipleBarGraphs(np.transpose(np.array(event_power_ratios_df)), 0.15, [0, 1], intervals,
                              error_values=np.transpose(np.array(event_sems_df)))


def PowerBandRatios(power_ratios_df):
    ## Get band ratio features
    theta_beta_ratios = {}  # Keys will be chan_theta_beta
    # Iterate through rows of power_ratios_df
    for i, row in power_ratios_df.iterrows():
        for ch in eeg_chans:
            curr_key = ch + "_theta_beta"
            if curr_key not in theta_beta_ratios:
                theta_beta_ratios[curr_key] = []

            # Calculate band ratios and append to dictionary
            power_bin_theta_key = ch + "_(4, 7)"
            power_bin_beta_key = ch + "_(12, 30)"
            theta_val = row[power_bin_theta_key]
            beta_val = row[power_bin_beta_key]
            theta_beta_ratios[curr_key].append(theta_val / beta_val)

    # Create df for band ratios: band_ratios_df
    band_ratios_df = pd.DataFrame(theta_beta_ratios)

    # Concatenate power_ratios_df with band_ratios_df to get full feature df
    feature_df = pd.concat([power_ratios_df, band_ratios_df], axis=1)

    ## Get channel ratio features
    # Similar algorithm as Band Ratios, but for each interval
    C3_C4_differences = {}
    for i, row in power_ratios_df.iterrows():
        for interval in intervals:
            curr_key = "C3_C4_diff_" + str(interval)
            if curr_key not in C3_C4_differences:
                C3_C4_differences[curr_key] = []

            # Calculate band ratios and append to dictionary
            power_bin_C3_key = "C3_" + str(interval)
            power_bin_C4_key = "C4_" + str(interval)

            C3_val = row[power_bin_C3_key]
            C4_val = row[power_bin_C4_key]

            C3_C4_differences[curr_key].append(C3_val - C4_val)

    # Create df for band ratios: band_ratios_df
    C3_C4_differences_df = pd.DataFrame(C3_C4_differences)
    # Concatenate power_ratios_df with band_ratios_df to get full feature df
    feature_df = pd.concat([power_ratios_df, band_ratios_df, C3_C4_differences_df], axis=1)


def neuroDSP_alpha_instantaneous_amplitude_median(W1_feature_df, eeg_epoch_full_df):
    alpha_range = (7, 12)
    alpha_amps = {}
    for i in range(0, len(eeg_epoch_full_df)):
        for ch in eeg_chans:
            amp = amp_by_time(eeg_epoch_full_df[ch][i][:], eeg_fs, alpha_range)
            key = ch + "_" + str(alpha_range) + "_inst_med"
            if key not in alpha_amps:
                alpha_amps[key] = list()
            alpha_amps[key].append(np.nanmedian(amp))

    alpha_med_df = pd.DataFrame(alpha_amps)
    feature_df = pd.concat([W1_feature_df, alpha_med_df], axis=1)
    return feature_df


def FOOOF():
    # Import required code for visualizing example models
    from fooof import FOOOF
    from fooof.sim.gen import gen_power_spectrum
    from fooof.sim.utils import set_random_seed
    from fooof.plts.annotate import plot_annotated_model

    # Set random seed, for consistency generating simulated data
    set_random_seed(10)

    # Simulate example power spectra
    freqs1, powers1 = gen_power_spectrum([3, 40], [1, 1],
                                         [[10, 0.2, 1.25], [30, 0.15, 2]])

    # Initialize power spectrum model objects and fit the power spectra
    fm1 = FOOOF(min_peak_height=0.05, verbose=False)
    fm1.fit(freqs1, powers1)

    plot_annotated_model(fm1, annotate_aperiodic=True)