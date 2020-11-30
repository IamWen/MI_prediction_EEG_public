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


# PSD plotting
def plotPSD(freq, psd, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    '''
    Inputs
    - freq: the list of frequencies corresponding to the PSDs
    - psd: the list of psds that represent the power of each frequency
    - pre_cut_off_freq: the lowerbound of the frequencies to show
    - post_cut_off_freq: the upperbound of the frequencies to show
    - label: a text label to assign this plot (in case multiple plots want to be drawn)

    Outputs:
    - None, except a plot will appear. plot.show() is not called at the end, so you can call this again to plot on the same axes.
    '''
    # Label the axes
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('log(PSD)')

    # Calculate the frequency point that corresponds with the desired cut off frequencies
    pre_cut = int(len(freq) * (pre_cut_off_freq / freq[-1]))
    post_cut = int(len(freq) * (post_cut_off_freq / freq[-1]))

    # Plot
    plt.plot(freq[pre_cut:post_cut], np.log(psd[pre_cut:post_cut]), label=label)


# Get Frequencies and PSDs from EEG data - this is the raw PSD method.
def getFreqPSDFromEEG(eeg_data, fs=eeg_fs):
    # Use scipy's signal.periodogram to do the conversion to PSDs
    freq, psd = signal.periodogram(eeg_data, fs=int(fs), scaling='spectrum')
    return freq, psd

# Get Frequencies and mean PSDs from EEG data - this yeilds smoother PSDs because it averages the PSDs made from sliding windows.
def getMeanFreqPSD(eeg_data, fs=eeg_fs):
    freq_mean, psd_mean = compute_spectrum(eeg_data, fs, method='welch', avg_type='mean', nperseg=fs*2)
    return freq_mean, psd_mean

# Plot PSD from EEG data (combines the a PSD calculator function and the plotting function)
def plotPSD_fromEEG(eeg_data, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    freq, psd = getMeanFreqPSD(eeg_data, fs=fs)
    plotPSD(freq, psd, fs, pre_cut_off_freq, post_cut_off_freq, label)

# Spectrogram plotting
def plotSpectrogram_fromEEG(eeg_data, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120):
    f, t, Sxx = signal.spectrogram(eeg_data, fs=fs)
    # Calculate the frequency point that corresponds with the desired cut off frequencies
    pre_cut = int(len(f) * (pre_cut_off_freq / f[-1]))
    post_cut = int(len(f) * (post_cut_off_freq / f[-1]))
    plt.pcolormesh(t, f[pre_cut:post_cut], Sxx[pre_cut:post_cut], shading='gouraud')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (sec)")


def plot_continuous_data_by_timepoint(eeg_df):
    # Try adjust these variables to see different time ranges!
    # A single trial is 4 seconds or 1000 timpoints (4 ms per timepoint)
    # Hint: refer to the Epoched data dataframe for the time of each trial
    start_time_ms = 223556  # Start time in millis
    start_time_timepoints = start_time_ms // 4  # Divide by 4 to get into timepoints
    end_time_timepoints = start_time_timepoints + 1000  # Specify number of more timepoints we want past start

    # Plot a single EEG channel
    plt.figure(figsize=(15, 5))
    plt.plot(eeg_df['C3'].values[start_time_timepoints:end_time_timepoints])
    plt.title("C3 -- " + str(start_time_timepoints) + " to " + str(end_time_timepoints))
    plt.xlabel("timepoints")
    plt.ylabel("Voltage (uV)")
    plt.show()

    # Plot a single EOG channel
    plt.figure(figsize=(15, 5))
    plt.plot(eeg_df['EOG:ch01'].values[start_time_timepoints:end_time_timepoints])
    plt.title("EOG:ch01 -- " + str(start_time_timepoints) + " to " + str(end_time_timepoints))
    plt.xlabel("timepoints")
    plt.ylabel("Voltage (uV)")
    plt.show()

    # Plot the PSD of the single EEG channel
    plt.figure(figsize=(15, 5))
    plotPSD_fromEEG(eeg_df['C3'].values[start_time_timepoints:end_time_timepoints], pre_cut_off_freq=2,
                    post_cut_off_freq=30, label="C3")
    plt.title("PSD of C3 in the timespan provided")
    plt.legend()
    plt.show()

    # Plot the spectrogram of the single EEG channel
    plt.figure(figsize=(15, 5))
    plotSpectrogram_fromEEG(eeg_df['C3'].values[start_time_timepoints:end_time_timepoints], pre_cut_off_freq=2,
                            post_cut_off_freq=30)
    plt.title("Spectrogram of C3 in the timespan provided")
    plt.show()


def plot_1_trial(eeg_epoch_full_df):
    # Visualize EEG and PSD for one trial
    # Try changing trial_num to view different trials!
    trial_num = 0
    eeg_chans = ["C3", "Cz", "C4"]  # 10-20 system
    eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]

    plt.figure(figsize=(15, 5))
    for ch in eeg_chans:
        plt.plot(eeg_epoch_full_df[ch][trial_num], label=ch)
    plt.ylabel("Voltage (uV)")
    plt.xlabel("timepoints @ 250Hz")
    plt.title("EEG of one motor imagery trial")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 5))
    for ch in eog_chans:
        plt.plot(eeg_epoch_full_df[ch][trial_num], label=ch)
    plt.ylabel("Voltage (uV)")
    plt.xlabel("timepoints @ 250Hz")
    plt.title("EOG of one motor imagery trial")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 5))
    for ch in eeg_chans:
        plotPSD_fromEEG(eeg_epoch_full_df[ch][trial_num], pre_cut_off_freq=2, post_cut_off_freq=30, label=ch)
    plt.title("PSD of EEG in one motor imagery trial")
    plt.legend()
    plt.show()


def get_PSD_avg(eeg_epoch_full_df):
    # Get PSD averages for each channel for each event type (0=left or 1=right)
    eeg_chans = ["C3", "Cz", "C4"]  # 10-20 system
    eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
    event_types = {0: "left", 1: "right"}

    psd_averages_by_type = {}
    for event_type in event_types.keys():
        psds_only_one_type={}
        freqs_only_one_type={}
        for i, row in eeg_epoch_full_df[eeg_epoch_full_df["event_type"] == event_type].iterrows():
            for ch in eeg_chans:
                if ch not in psds_only_one_type:
                    psds_only_one_type[ch] = list()
                    freqs_only_one_type[ch] = list()
                f, p = getMeanFreqPSD(row[ch])
                psds_only_one_type[ch].append(p)
                freqs_only_one_type[ch].append(f)
        avg_psds_one_type = {}
        for ch in eeg_chans:
            psds_only_one_type[ch] = np.array(psds_only_one_type[ch])
            avg_psds_one_type[ch] = np.mean(psds_only_one_type[ch], axis=0)
        psd_averages_by_type[event_type] = dict(avg_psds_one_type)

    # View Average PSDs
    for event_type in event_types.keys():
        for ch in eeg_chans[:]:
            plotPSD(freqs_only_one_type[eeg_chans[0]][0], psd_averages_by_type[event_type][ch], pre_cut_off_freq=2,
                    post_cut_off_freq=30, label=ch)

        plt.legend()
        plt.title("event type: " + event_types[event_type])
        plt.show()



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


def plot_avg_PowerRatios():
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


def getPowerBandRatios(power_ratios_df):
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
    return power_ratios_df, feature_df


def get_channel_relationships(power_ratios_df, band_ratios_df):
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
    alpha_pha = {}
    alpha_if = {}

    for i in range(0, len(eeg_epoch_full_df)):
        for ch in eeg_chans:
            sig = eeg_epoch_full_df[ch][i][:]
            key = ch + "_" + str(alpha_range) + "_inst_med"

            amp = amp_by_time(sig, eeg_fs, alpha_range) # Amplitude by time (instantaneous amplitude)
            if key not in alpha_amps:
                alpha_amps[key] = list()
            alpha_amps[key].append(np.nanmedian(amp))

            pha = phase_by_time(sig, eeg_fs, alpha_range) # Phase by time (instantaneous phase)
            if key not in alpha_pha:
                alpha_pha[key] = list()
            alpha_pha[key].append(np.nanmedian(pha))

            i_f = freq_by_time(sig, eeg_fs, alpha_range) # Frequency by time (instantaneous frequency)
            if key not in alpha_if:
                alpha_if[key] = list()
            alpha_if[key].append(np.nanmedian(i_f))


    alpha_med_df = pd.DataFrame(alpha_amps)
    alpha_pha_df = pd.DataFrame(alpha_pha)
    alpha_if_df = pd.DataFrame(alpha_if)
    feature_df = pd.concat([W1_feature_df, alpha_med_df, alpha_pha_df,alpha_if_df], axis=1)
    return feature_df


def FOOOF(eeg_epoch_full_df, W1_feature_df):
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
    # Constants
    freq_range = [1, 40]
    alpha_range = (7, 12)
    # Initialize a FOOOF object
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.4)

    # Get the PSD of our EEG Signal
    sig = eeg_epoch_full_df['Cz'][0]
    freq, psd = getMeanFreqPSD(sig)

    fm.add_data(freq, psd, freq_range)

    fm.fit()
    fm.report()

    # Get PSD averages for each channel for each event type (0=left or 1=right)
    psd_averages_by_type = {}

    for event_type in event_types.keys():
        psds_only_one_type = {}
        freqs_only_one_type = {}
        for i, row in eeg_epoch_full_df[eeg_epoch_full_df["event_type"] == event_type].iterrows():
            for ch in eeg_chans:
                if ch not in psds_only_one_type:
                    psds_only_one_type[ch] = list()
                    freqs_only_one_type[ch] = list()
                f, p = getMeanFreqPSD(row[ch])
                psds_only_one_type[ch].append(p)
                freqs_only_one_type[ch].append(f)
        avg_psds_one_type = {}
        for ch in eeg_chans:
            psds_only_one_type[ch] = np.array(psds_only_one_type[ch])
            avg_psds_one_type[ch] = np.mean(psds_only_one_type[ch], axis=0)
        psd_averages_by_type[event_type] = dict(avg_psds_one_type)

    # Visualize the parameters in these two classes of C4 activity
    fm_left_C4 = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.4)
    fm_left_C4.add_data(freqs_only_one_type[eeg_chans[0]][0], psd_averages_by_type[0]['C4'], freq_range)
    fm_left_C4.fit()
    fm_left_C4.report()

    fm_right_C4 = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.4)
    fm_right_C4.add_data(freqs_only_one_type[eeg_chans[0]][0], psd_averages_by_type[1]['C4'], freq_range)
    fm_right_C4.fit()
    fm_right_C4.report()


    # Calculate central freq, alpha power, and bandwidth for each channel and each trial
    # This cell takes a few minutes to run (~8 mins on my computer). There are 3680 trials in the training data.

    # Initialize a fooof object
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.4)

    # Some will not have alpha peaks, use these variables to keep track
    num_with_alpha = 0
    num_without_alpha = 0
    fooof_parameters = {}
    for i in range(len(eeg_epoch_full_df)):
        # Print the trial number every 100 to make sure we're making progress
        if i % 100 == 0:
            print(i)
        for ch in eeg_chans:
            # Determine the key
            CF_key = ch + "_alpha_central_freq"
            PW_key = ch + "_alpha_power"
            BW_key = ch + "_alpha_band_width"
            if CF_key not in fooof_parameters:
                fooof_parameters[CF_key] = []
            if PW_key not in fooof_parameters:
                fooof_parameters[PW_key] = []
            if BW_key not in fooof_parameters:
                fooof_parameters[BW_key] = []

            # Calculate the PSD for the desired signal
            sig = eeg_epoch_full_df[ch][i]
            freq, psd = getMeanFreqPSD(sig)

            # Set the frequency and spectral data into the FOOOF model and get peak params
            fm.add_data(freq, psd, freq_range)
            fm.fit()
            peak_params = fm.peak_params_

            # Only select the peaks within alpha power
            peak_params_alpha = []
            for param in peak_params:
                if (param[0] > alpha_range[0]) and (param[0] < alpha_range[1]):
                    peak_params_alpha.append(param)

            # Take the average if there are multiple peaks detected, otherwise 0 everything
            means = []
            if len(peak_params_alpha) > 0:
                num_with_alpha += 1
                means = np.mean(peak_params_alpha, axis=0)
            else:
                num_without_alpha += 1
                means = [0, 0, 0]

            fooof_parameters[CF_key].append(means[0])
            fooof_parameters[PW_key].append(means[1])
            fooof_parameters[BW_key].append(means[2])

    # Concatenate
    fooof_parameters_df = pd.DataFrame(fooof_parameters)

    feature_df = pd.concat([W1_feature_df, fooof_parameters_df], axis=1)
    print("% with alpha:", num_with_alpha / (num_with_alpha + num_without_alpha))

    # Save it so you don't need to spend extra time rerunning the heavy computation cell from above.
    feature_df.to_pickle("W2_feature_df.pkl")