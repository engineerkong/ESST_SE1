"""
This file uses boxplot to demonstrate the outcome episode length in experiment 01 to evaluate RL algorithms and reward functions.
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
res_sim = []
res_real = []
for i in range(len(t)):
    tmp_sim = []
    tmp_real = []
    for target_file in t[i]:
        df = pandas.read_csv(target_file)
        sim = json.loads(df["sim"].iloc[0])
        len_sim = sim["len"]
        tmp_sim += len_sim
        real = json.loads(df["real"].iloc[0])
        len_real = real["len"]
        tmp_real += len_real
    res_sim.append(tmp_sim)
    res_real.append(tmp_real)
fig, ax = plt.subplots()
fig.set_size_inches(9,4.5)
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

plt.title('The Episode Lengths of RL Models with various Algorithms and Reward Functions',fontsize=12)
plt.xlabel('RL Models',fontsize=12)
plt.ylabel('Episode Length',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks([1.15, 2.15, 3.15, 4.15, 5.15, 6.15], ['sac gym', 'ppo gym', 'sac mygym', 'ppo mygym', 'sac compo', 'ppo compo'],fontsize=12)

# 定义自定义的图例
plt.legend([box1["boxes"][0], box2["boxes"][0]], ['SE1', 'SE2'], loc='upper right')

# 显示图表
plt.show()