import pandas
import utilities
import os
import matplotlib.pyplot as plt


data_path = "data/"
csvs_path = data_path + "oneyeardata/"
year_csv_path = data_path + "year_data.csv"
norm_year_path = data_path + "norm_year_data.csv"
graphs_path = "graphs/"
visualize_data = False

if not os.path.exists(year_csv_path):
    year_dataframe = utilities.get_year_dataframe(csvs_path)
    year_dataframe.to_csv(year_csv_path, sep=",", index=False)
else:
    year_dataframe = pandas.read_csv(year_csv_path)

norm_columns = year_dataframe.columns[2:].tolist()
if not os.path.exists(norm_year_path):
    norm_year_df = utilities.normalize_data(year_dataframe, norm_columns)
    norm_year_df.to_csv(norm_year_path, sep=",", index=False)
else:
    norm_year_df = pandas.read_csv(norm_year_path)

if visualize_data:

    fig, axs = plt.subplots(8, 1, sharex='all', sharey='all', figsize=(19.20, 19.20))
    for i, col in enumerate(norm_columns):

        axs[i].plot(norm_year_df[col])
        axs[i].set_title(col)

    plt.tight_layout()
    plt.savefig(graphs_path + "norm_data.pdf")
    plt.show()

print(norm_year_df.iloc[0].values)
