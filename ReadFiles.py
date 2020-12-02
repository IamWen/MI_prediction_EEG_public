import pandas as pd # For working with DataFrames
import numpy as np # For ease of array manipulation + basic stats
import matplotlib.pyplot as plt # For plotting pretty plots :)
from scipy import signal # For calculating PSDs and plotting spectrograms

from neurodsp.spectral import compute_spectrum # for smoothed PSD computation
from pathlib import Path # For making paths compatible on Windows and Macs


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


def read_raw_data_all(dir):
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


def read_raw_data_one(dir):
    # Load a subject's data
    filename = "B0101T"
    eeg_filename = Path(dir + '/train/' + filename + ".csv")
    event_filename = Path(dir+'/y_train_only/' + filename + ".csv")

    eeg_chans = ["C3", "Cz", "C4"]  # 10-20 system
    eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
    all_chans = eeg_chans + eog_chans
    event_types = {0: "left", 1: "right"}

    # Load the raw csvs into dataframes
    eeg_df = pd.read_csv(eeg_filename)
    event_df = pd.read_csv(event_filename)

    print("recording length:", eeg_df["time"].values[-1] / 1000 / 60, "min")
    return eeg_df, event_df


def read_epoched_data(dir):
    # We've already epoched all the data into 4000ms trials for you in epoched_train.pkl and epoched_test.pkl :)
    # These are the epochs that will be used in accuracy evaluation
    fname_tr = '/epoched_train.pkl'
    fname_ts = 'epoched_test.pkl'
    fname_w1 = '/W1_feature_df.pkl'
    fname_w2 = '/W2_feature_df.pkl'
    epoch_df_filename = Path(dir+fname_tr)
    eeg_epoch_full_df = pd.read_pickle(epoch_df_filename)
    W1 = pd.read_pickle(Path(dir+fname_w1))
    W2 = pd.read_pickle(Path(dir+fname_w2))
    print(list(W1.columns))
    return eeg_epoch_full_df, W1,W2


if __name__ == "__main__":
    dir = 'E:/USC/EE660_2020/data'
    eeg_epoch_full_df, W1,W2 = read_epoched_data(dir)
    y = W1[['y']]
    X = W1[W1.columns[~W1.columns.isin(['y'])]]

    print('done')