import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dir_data = '../at_home_data'

df_train_rec = []

for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path):
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs
                df_train_rec.append(df)

df_all = pd.concat(df_train_rec, ignore_index=True)

block_size = 25

df_all = df_all.sort_values(
    ["subject_id", "session_num", "session_part", "trial"]
).reset_index(drop=True)

df_all['acc'] = (df_all['cat'] == df_all['resp']).astype(int)
df_all['trial'] = df_all.groupby(['subject_id', 'session_num']).cumcount()
df_all['n_trials'] = df_all.groupby(['subject_id', 'session_num'])['trial'].transform('count')
df_all['block'] = df_all.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

dd = (
    df_all.groupby(["subject_id", "session_num", "block"], as_index=False)["acc"]
    .mean()
    .sort_values(["session_num", "subject_id", "block"])
)

days = dd["session_num"].unique()[:8]  # no sorted()

fig, axes = plt.subplots(1, 8, figsize=(24, 3.5), sharey=True)

for ax, day in zip(axes, days):
    sns.lineplot(
        data=dd[dd["session_num"] == day],
        x="block",
        y="acc",
        hue="subject_id",
        units="subject_id",
        estimator=None,
        legend=False,
        alpha=0.5,
        lw=1,
        ax=ax,
    )
    ax.set_title(f"Day {day}")
    ax.set_xlabel("Block")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Accuracy")
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()
plt.show()

dd_day = dd.groupby(['subject_id', 'session_num']).agg({'acc': 'mean'}).reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False)
sns.pointplot(data=dd_day, x='session_num', y='acc', errorbar=('se'), ax=ax[0,0])
plt.tight_layout()
plt.show()


