"""
This file uses boxplot to demonstrate the outcome episode length in experiment 02 to evaluate hyperparameters.
Kong/29.04.2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import json

folder_path_1 = "/home/lingxiao/Desktop/evaluating_models/gym_sac/"
folder_path_2 = "/home/lingxiao/Desktop/evaluating_models/gym_ppo/"
folder_path_3 = "/home/lingxiao/Desktop/evaluating_models/mygym_sac/"
folder_path_4 = "/home/lingxiao/Desktop/evaluating_models/mygym_ppo/"
folder_path_5 = "/home/lingxiao/Desktop/evaluating_models/composuite_sac/"
folder_path_6 = "/home/lingxiao/Desktop/evaluating_models/composuite_ppo/"
file_name = "data.csv"
def search_files_by_name(folder_path, file_name):
    target_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                target_files.append(file_path)
    return target_files
target_files_1 = search_files_by_name(folder_path_1,file_name)
target_files_2 = search_files_by_name(folder_path_2,file_name)
target_files_3 = search_files_by_name(folder_path_3,file_name)
target_files_4 = search_files_by_name(folder_path_4,file_name)
target_files_5 = search_files_by_name(folder_path_5,file_name)
target_files_6 = search_files_by_name(folder_path_6,file_name)
t = [target_files_1,target_files_2,target_files_3,target_files_4,target_files_5,target_files_6]
mean = []
std = []
for i in range(len(t)):
    data = []
    for target_file in t[i]:
        df = pandas.read_csv(target_file)
        sim = json.loads(df["sim"].iloc[0])
        iqm_sim = sim["iqm"]
        data += iqm_sim
        real = json.loads(df["real"].iloc[0])
        iqm_real = real["iqm"]
        data += iqm_real
    data = np.array(data)
    mean.append(np.mean(data))
    std.append(np.std(data))
print(mean, std)
res_sim = []
res_real = []
for i in range(len(t)):
    tmp_sim = []
    tmp_real = []
    for target_file in t[i]:
        df = pandas.read_csv(target_file)
        sim = json.loads(df["sim"].iloc[0])
        iqm_sim = sim["iqm"]
        for j in range(len(iqm_sim)):
            iqm_sim[j] = (iqm_sim[j] - mean[i])/std[i]
        tmp_sim += iqm_sim
        real = json.loads(df["real"].iloc[0])
        iqm_real = real["iqm"]
        for j in range(len(iqm_real)):
            iqm_real[j] = (iqm_real[j] - mean[i])/std[i]
        tmp_real += iqm_real
    res_sim.append(tmp_sim)
    res_real.append(tmp_real)
fig, ax = plt.subplots()
box1 = ax.boxplot(res_sim, positions=[1,2,3,4,5,6], widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='green', linewidth=1.5),
                  capprops=dict(color='red', linewidth=1.5),
                  medianprops=dict(color='yellow', linewidth=0),
                  flierprops=dict(marker='None', color='black', markerfacecolor='red'))
box2 = ax.boxplot(res_real, positions=[1.3,2.3,3.3,4.3,5.3,6.3], widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='pink', color='red'),
                  whiskerprops=dict(color='blue', linewidth=1.5),
                  capprops=dict(color='navy', linewidth=1.5),
                  medianprops=dict(color='green', linewidth=0),
                  flierprops=dict(marker='None', color='black', markerfacecolor='pink'))
print("Boxes:")
for box in box2['boxes']:
    path = box.get_path()  # 获取 Path 对象
    print(path.vertices[:, 1])  # 打印 y 坐标数据

plt.title('The Episode Lengths of RL Models with various Algorithm and Reward Function Combinations')
plt.xlabel('RL Models')
plt.ylabel('Episode Length')
plt.xticks([1.15, 2.15, 3.15, 4.15, 5.15, 6.15], ['sac gym', 'ppo gym', 'sac mygym', 'ppo mygym', 'sac composuite', 'ppo composuite'])

# 定义自定义的图例
plt.legend([box1["boxes"][0], box2["boxes"][0]], ['SE1', 'SE2'], loc='upper right')

# 显示图表
plt.show()