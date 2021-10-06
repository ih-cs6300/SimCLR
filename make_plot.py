# create a plot from log file data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./simclr_log.csv")
exp1_df = df.iloc[:125, :]
exp2_df = df.iloc[125:, :]

exp1_df['acc1'] = exp1_df['acc1'] / 100.
exp2_df['acc1'] = exp2_df['acc1'] / 100.

exp1_groups = exp1_df.groupby('nshots')
exp2_groups = exp2_df.groupby('train_size')

# nshots = []
# mu = []
# std = []
# for gname, group in exp1_groups:
#     mu.append(np.mean(group)['acc1'].item())
#     std.append(np.std(group)['acc1'].item())
#     nshots.append(gname)
#
# # move -1 and it's value to the end
# mu.append(mu.pop(0))
# nshots.pop(0)
# nshots.append(32000) # all the training data 0.7 * 45848
# std.append(std.pop(0))
#
# # kmeans feature extraction data
# kmeans_df = pd.read_csv("../kmeans_feat_extract/kmeans_feat_log.csv")
#
# x = kmeans_df['nshots'].tolist()
# y = kmeans_df['acc1'].tolist()
#
# plt.plot(x, y, marker='o', linestyle='-', color='r', markersize='4')
# plt.plot(nshots, mu, color='g', marker='x')
# plt.ylabel("Accuracy")
# plt.xlabel("Num shots")
# plt.grid(color='k', linestyle='-', linewidth=1)
# plt.title("Shots vs. Accuracy")
# plt.legend(["Kmeans", "SimCLR"])
#plt.show()
#plt.savefig("shotvacc_plot.png")


#########################################################################################
# experiment 2

trainset_size = []
ts_mu = []
ts_std = []
for gname, group in exp2_groups:
    ts_mu.append(np.mean(group)['acc1'].item())
    ts_std.append(np.std(group)['acc1'].item())
    trainset_size.append(gname)

# move -1 and it's value to the end
ts_mu.append(ts_mu.pop(0))
trainset_size.pop(0)
trainset_size.append(32000) # all the training data 0.7 * 45848
ts_std.append(ts_std.pop(0))

#plt.plot(trainset_size, ts_mu, marker='o', linestyle='-', color='r', markersize='4')
plt.errorbar(trainset_size, ts_mu, yerr=ts_std, color='r')
plt.ylabel("Accuracy")
plt.xlabel("Training Set Size")
plt.grid(color='k', linestyle='-', linewidth=1)
plt.title("SimCLR: Training Set Size vs. Accuracy")
#plt.show()

exp3_df = pd.read_csv("./simclr_log1.csv")
exp3_df['acc1'] = exp3_df['acc1'] / 100.

exp3_df = exp3_df.iloc[44:50, :]
x_exp3 = exp3_df['train_size'].tolist()
y_exp3 = exp3_df['acc1'].tolist()
x_exp3.pop(-1)
x_exp3.append(32000)
plt.legend()

plt.plot(x_exp3, y_exp3, color='green', marker='^')
plt.legend(["nshots=10", "nshots = 128"])
#plt.show()
plt.savefig("trainsetsizevacc_plot.png")
