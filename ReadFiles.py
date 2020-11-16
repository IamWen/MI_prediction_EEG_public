import pandas as pd # For working with DataFrames
import numpy as np # For ease of array manipulation + basic stats
import matplotlib.pyplot as plt # For plotting pretty plots :)
from scipy import signal # For calculating PSDs and plotting spectrograms

from neurodsp.spectral import compute_spectrum # for smoothed PSD computation
from pathlib import Path # For making paths compatible on Windows and Macs

import ReadFiles as RF


eeg_fs = 250 # Data was recorded at 250 Hz


## Create DF for each of these, columns are channels, each row is a trial run
def getDF(epochs, labels, times, chans):
    data_dict = {}
    for i, label in enumerate(labels):
        start_time = times[i][0]
        if 'start_time' not in data_dict:
            data_dict['start_time'] = list()
        data_dict['start_time'].append(start_time)

        if 'event_type' not in data_dict:
            data_dict['event_type'] = list()
        data_dict['event_type'].append(label)

        for ch in range(len(chans)):
            if chans[ch] not in data_dict:
                data_dict[chans[ch]] = list()
            data_dict[chans[ch]].append(epochs[i][ch])

    return pd.DataFrame(data_dict)


# Extract data from raw dataframes for constructing trial-by-trial dataframe
def getEpochedDF(eeg_df, event_df, trial_duration_ms=4000):
    epochs = []
    epoch_times = []
    labels = []
    start_df = eeg_df[eeg_df['EventStart'] == 1]
    for i, event_type in enumerate(event_df["EventType"].values):
        labels.append(event_type)
        start_time = start_df.iloc[i]["time"]
        end_time = int(start_time + trial_duration_ms)
        epoch_times.append((start_time, end_time))
        sub_df = eeg_df[(eeg_df['time'] > start_time) & (eeg_df['time'] <= end_time)]
        eeg_dat = []
        for ch in all_chans:
            eeg_dat.append(sub_df[ch].values)
        epochs.append(np.array(eeg_dat))

    # Create dataframe from the data extracted previously
    eeg_epoch_df = getDF(epochs, labels, epoch_times, all_chans)
    return eeg_epoch_df


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



def read_raw_data(dir):
    import os
    X_list, y_list = [],[]
    n, labels = 0, 0
    for one_file in os.listdir(dir+'/train'):
        # print(one_file)
        if 'csv' in one_file:
            x_fname = dir + '/train/' + one_file
            print(x_fname)

            # Load a subject's data
            eeg_filename = Path(x_fname)
            event_filename = Path(dir+'/y_train_only/'+one_file)

            eeg_chans = ["C3", "Cz", "C4"]  # 10-20 system
            eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
            all_chans = eeg_chans + eog_chans
            event_types = {0: "left", 1: "right"}

            # Load the raw csvs into dataframes
            eeg_df = pd.read_csv(eeg_filename)
            event_df = pd.read_csv(event_filename)
            # print("recording length:", eeg_df["time"].values[-1] / 1000 / 60, "min")

            X_list.append(eeg_df)
            y_list.append(event_df)
            print('length:' + str(event_df.size))
            n += event_df.size
            label = event_df.sum()
            labels += label
    return X_list, y_list, n, labels


def read_epoched_data(dir):
    # We've already epoched all the data into 4000ms trials for you in epoched_train.pkl and epoched_test.pkl :)
    # These are the epochs that will be used in accuracy evaluation
    fname_tr = '/epoched_train.pkl'
    fname_ts = 'epoched_test.pkl'
    epoch_df_filename = Path(dir+fname_tr)
    eeg_epoch_full_df = pd.read_pickle(epoch_df_filename)
    eeg_epoch_full_df.head(2)


if __name__ == "__main__":
    dir = 'E:/USC/EE660_2020/data'

    # read_epoched_data(dir)
    X_list, y_list, n, labels = read_raw_data(dir)
    print(n)
    print(labels)
    print('done')