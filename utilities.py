import pandas
import os
import numpy as np
import sklearn.preprocessing as skl_prep
import datetime
import pynever.networks as pyn_networks
import pynever.strategies.conversion as pyn_conv
import torch


def parse_csv_name(name: str):
    year = 2018
    month = (9 + int(name[0:2]) - 1) % 12 + 1  # needed since the first month is actually october
    if 1 <= month < 10:
        year = 2019
    day = int(name[3:5])
    hour = int(name[6:8])
    minute = int(name[8:10])
    second = int(name[10:12])
    mode = int(name[21])
    starting_datetime = datetime.datetime(year, month, day, hour, minute, second)
    pandas_s_datetime = pandas.to_datetime(starting_datetime)
    return mode, pandas_s_datetime


def get_year_dataframe(csvs_path: str):
    csv_ids = sorted(os.listdir(csvs_path))
    dataframe_list = []
    for csv_id in csv_ids:
        temp_dataframe = pandas.read_csv(csvs_path + csv_id)
        mode, start_datetime = parse_csv_name(csv_id)
        mode_list = np.array([mode for _ in range(temp_dataframe.__len__())])
        for i in range(temp_dataframe.__len__()):
            new_timestamp = start_datetime + pandas.to_timedelta(temp_dataframe['timestamp'][i], "seconds")
            temp_dataframe.loc[i, 'timestamp'] = new_timestamp

        temp_dataframe.insert(1, 'mode', mode_list)

        dataframe_list.append(temp_dataframe)

    year_dataframe = pandas.concat(dataframe_list, ignore_index=True)
    return year_dataframe


def normalize_data(df: pandas.DataFrame, columns: list):
    scalers = {}
    for c in columns:
        scalers[c] = skl_prep.MinMaxScaler((-1, 1)).fit(df[c].values.reshape(-1, 1))

    norm_df = df.copy()
    for c in columns:
        norm = scalers[c].transform(norm_df[c].values.reshape(-1, 1))
        norm_df[c] = norm

    return norm_df


def compute_loss(network: pyn_networks, loss_f, sample, target):

    pyt_net = pyn_conv.PyTorchConverter().from_neural_network(network).pytorch_network
    pyt_net.to("mps")
    with torch.no_grad():
        pyt_net.float()
        sample = sample.float()
        target = target.float()
        sample, target = sample.to("mps"), target.to("mps")
        outputs = pyt_net(sample)
        loss = loss_f(outputs, target)
    return loss




