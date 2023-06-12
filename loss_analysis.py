import pandas
import matplotlib.pyplot as plt


exp_folder = "exp_001"
exp_info = pandas.read_csv(f"{exp_folder}/logs/models_gen_logs.csv")
model_ids = exp_info['net_id'].values
original_df = pandas.read_csv(f"data/norm_year_data.csv")
month_ids = ["-10-", "-11-", "-12-", "-01-", "-02-", "-03-", "-04-", "-05-", "-06-", "-07-", "-08-", "-09-"]
col_labels = ['Torque', 'Cut lag', 'Cut position', 'Cut speed', 'Film position', 'Film speed', 'Film lag', 'VAX']

save_format = ".png"

month_starts = []
start = 0
for month_id in month_ids:
    month_starts.append(start)
    start += original_df[original_df['timestamp'].str.contains(month_id)].__len__()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'orange']

fig, axs = plt.subplots(len(model_ids), 1, figsize=(12, 15))
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)
model_idx = 0
ae_colors = ['r', 'g', 'b', 'orange']
ae_labels = ['A1', 'A2', 'A3', 'A4']
ae_markers = ['o', 'v', 's', '*']
for model_id in model_ids:

    losses_df = pandas.read_csv(f"{exp_folder}/models_data/{model_id}_losses.csv")
    mean_losses = losses_df.mean(axis=1).values
    axs[model_idx].scatter(range(mean_losses.shape[0]), mean_losses, s=1, c=ae_colors[model_idx],
                           label=ae_labels[model_idx], marker=ae_markers[model_idx])
    axs[model_idx].set_xticks(month_starts, [i for i in range(1, 13)], fontsize="15")
    axs[model_idx].set_ylabel("MSE")
    axs[model_idx].set_ylim((0, 0.22))
    axs[model_idx].set_xlim((0, mean_losses.shape[0]))
    axs[model_idx].grid(axis='x', color='black')
    axs[model_idx].legend(markerscale=8, fontsize=15)
    model_idx += 1

axs[model_idx - 1].set_xlabel("Months")
plt.tight_layout()
plt.savefig(f"{exp_folder}/graphs/mean_losses_year{save_format}")
plt.show()