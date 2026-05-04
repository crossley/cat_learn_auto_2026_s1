import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dir_data = '../at_home_data'
dir_data_lab = '../behavioural_data'

df_train_rec = []
df_lab_rec = []

# not reading in cp task
for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and "task_cp_" not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs
                df_train_rec.append(df)

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path):

                # on day 1, participant 875 did 10 train trials and 60 probe trials
                # taking first 10 train trials and adding it to part 2 (540 train,
                # 100 probe) -- excluding for the moment
                if fs not in ['sub_875_sess_001_part_001_date_2026_04_03_data.csv',
                              'sub_875_sess_001_part_002_date_2026_04_03_data.csv']:

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

df_home = pd.concat(df_train_rec, ignore_index=True)
df_lab = pd.concat(df_lab_rec, ignore_index=True)

block_size = 25

# -- Home -- 
df_home = df_home.sort_values(
    ["subject_id", "session_num", "session_part", "trial"]).reset_index(drop=True)
df_home = df_home[df_home['session_num'] != 17]

df_home['acc'] = (df_home['cat'] == df_home['resp']).astype(int)
df_home['trial'] = df_home.groupby(['subject_id', 'session_num']).cumcount()
df_home['n_trials'] = df_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
df_home['block'] = df_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
df_home = df_home.drop(columns=['value_left', 'size_left', 'value_right',
                                'size_right', 'congruency', 'cue',
                                'resp_key_ns', 'resp_ns', 'fb_ns', 'rt_ns',
                                't_cue_ns', 't_fb_ns'])

dd_home = (
    df_home.groupby(["subject_id", "session_num", "block"], as_index=False)["acc"]
    .mean()
    .sort_values(["session_num", "subject_id", "block"])
)

days_home = dd_home["session_num"].unique()[:16]  # no sorted()

# -- Dual Task (17) --
df_dt = df_home.sort_values(
    ["subject_id", "session_num", "session_part", "trial"]).reset_index(drop=True)
df_dt = df_dt[df_dt['session_num'] == 17]

df_dt['acc'] = (df_dt['cat'] == df_home['resp']).astype(int)
df_dt['trial'] = df_dt.groupby(['subject_id', 'session_num']).cumcount()
df_dt['n_trials'] = df_dt.groupby(['subject_id', 'session_num'])['trial'].transform('count')
df_dt['block'] = df_dt.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# -- Lab --
df_lab = df_lab.sort_values(
    ["subject_id", "session_num", "session_part", "trial"]).reset_index(drop=True)

df_lab['acc'] = (df_lab['cat'] == df_lab['resp']).astype(int)
df_lab['trial'] = df_lab.groupby(['subject_id', 'session_num']).cumcount()
df_lab['n_trials'] = df_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
df_lab['block'] = df_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

df_lab_all = df_lab.groupby(["subject_id", "session_num", "block", "probe_condition"], as_index=False)["acc"] .mean().sort_values(["session_num", "subject_id", "block"])

df_lab_train = df_lab[df_lab['phase'] == 'train'].groupby(['subject_id',
                                                           'session_num',
                                                           'probe_condition',
                                                           'block']).agg({'acc':
                                                                          'mean'}).reset_index()
df_lab_test = df_lab[df_lab['phase'] == 'test'].groupby(['subject_id',
                                                         'session_num',
                                                         'probe_condition',
                                                         'block']).agg({'acc':
                                                                        'mean'}).reset_index()

days_lab = dd_lab["session_num"].unique()[:5]  # no sorted()

# -- Plots --
# HOME: Plot whole expt across all days 
fig, axes = plt.subplots(1, len(days_home), figsize=(24, 3.5), sharey=True)
for ax, day in zip(axes, days_home):
    sns.lineplot(
        data=dd_home[dd_home["session_num"] == day],
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

# HOME: Plot mean of each day
dd_home_day = dd_home.groupby(['subject_id', 'session_num']).agg({'acc': 'mean'}).reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False)
sns.pointplot(data=dd_home_day, x='session_num', y='acc', errorbar=('se'), ax=ax[0,0])
plt.tight_layout()
plt.show()

# LAB: Plot whole expt across all days
extra_blocks = df_lab_all.loc[df_lab_all['block'] > 26]

fig, axes = plt.subplots(1, 5, figsize=(24, 3.5), sharey=True)
for ax, day in zip(axes, days_lab):
    sns.lineplot(
        data=df_lab_all[df_lab_all["session_num"] == day],
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

# LAB: Plot each participants average accuracy per day
df_lab_pd_avg = df_lab_train.groupby(['subject_id', 'session_num']).agg({'acc': 'mean'}).reset_index()

days_lab = df_lab_pd_avg["session_num"].unique()[:5]

fig, axes = plt.subplots(1, len(days_lab), squeeze = False)
for axes, day in zip(axes, days_lab):
    sns.pointplot(
        data=df_lab_pd_avg[df_lab_pd_avg["session_num"] == day],
        x="subject_id",
        y="acc",
        estimator=None,
        legend=False,
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

fig, axes = plt.subplots(1, len(days_lab), squeeze=False)
axes = axes.flatten()

for ax, day in zip(axes, days_lab):
    dday = df_lab_pd_avg[df_lab_pd_avg["session_num"] == day]
    if dday.empty:
        ax.set_visible(False)
        continue

    sns.pointplot(
        data=dday,
        x="subject_id",
        y="acc",
        errorbar=None,
        ax=ax,
    )
    ax.set_title(f"Day {day}")
    ax.set_xlabel("Subject")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Accuracy")
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()
plt.show()

# LAB: Plot mean of each day
dd_lab_day = df_lab_train.groupby(['subject_id', 'session_num',
                             'probe_condition']).agg({'acc':
                                                      'mean'}).reset_index()

fig, ax = plt.subplots(1, 1, squeeze=False)
sns.pointplot(data=dd_lab_day, x='session_num', y='acc', errorbar=('se'), ax=ax[0,0])
plt.tight_layout()
plt.show()

# LAB: Plot cost (pre - post)
# Cost 25 trials prior to probe (block 21) vs 25 trials post probe (block 22)
pre_90 = dd_lab[(dd_lab['block']==21) &
                (dd_lab['probe_condition']==90)].groupby('session_num')['acc'].mean()
post_90 = dd_lab[dd_lab['block']==22 &
                 (dd_lab['probe_condition']==90)].groupby('session_num')['acc'].mean()

pre_180 = dd_lab[(dd_lab['block']==21) &
                 (dd_lab['probe_condition']==180)].groupby('session_num')['acc'].mean()
post_180 = dd_lab[dd_lab['block']==22 &
                  (dd_lab['probe_condition']==180)].groupby('session_num')['acc'].mean()

cost_90 = pre_90 - post_90
cost_180 = pre_180 - post_180

cost_total = pd.concat([cost_90.rename('cost_90'), cost_180.rename('cost_180')], axis=1)

# wide -> long for seaborn
cost_total = cost_total.rename_axis("session_num").reset_index()

cost_plot = cost_total.melt(
    id_vars='session_num',
    value_vars=['cost_90', 'cost_180'],
    var_name='probe_condition',
    value_name='cost'
)

cost_plot["probe_condition"] = cost_plot["probe_condition"].map(
    {"cost_90": "90", "cost_180": "180"}
)

days_lab = sorted(cost_plot["session_num"].unique())

fig, axes = plt.subplots(1, len(days_lab), figsize=(3.5 * len(days_lab), 3.5), sharey=True)

# handle case len(days_lab)==1
if len(days_lab) == 1:
    axes = [axes]

for ax, day in zip(axes, days_lab):
    dday = cost_plot[cost_plot["session_num"] == day]

    sns.barplot(
        data=dday,
        x="probe_condition",
        y="cost",
        ax=ax,
        errorbar=None,   # one value per bar here (since you averaged already)
    )
    ax.set_title(f"Session {day}")
    ax.set_xlabel("Probe")
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("Cost (Block 21 - Block 22)")
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()
plt.show()

