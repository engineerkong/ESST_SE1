"""
This file uses boxplot to demonstrate the outcome normalized reward in experiment 02 to evaluate RL hyperparameters.
Kong/29.04.2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import json

folder_path = "/home/lingxiao/Desktop/evaluating_models/gym_sac_hps/"
file_name = "data.csv"
def search_files_by_name(folder_path, file_name):
    target_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                target_files.append(file_path)
    return target_files
target_files = search_files_by_name(folder_path,file_name)
data = []
for target_file in target_files:
    df = pandas.read_csv(target_file)
    sim = json.loads(df["sim"].iloc[0])
    data += sim["iqm"]
    real = json.loads(df["real"].iloc[0])
    len_real = real["iqm"]        
    data += real["iqm"]
data = np.array(data)
mean = np.mean(data)
std = np.std(data)
dic1 = {}
dic2 = {}
for target_file in target_files:
    df = pandas.read_csv(target_file)
    lr = df["ls.learning_rate"].iloc[0]
    gm = df["ls.gamma"].iloc[0]
    string = 'lr:' + str(round(lr,4)) + '\n'+ 'gm:' + str(round(gm,4))
    if string not in dic1.keys():
        dic1[string] = []
        dic2[string] = []
    sim = json.loads(df["sim"].iloc[0])
    iqm_sim = sim["iqm"]
    for j in range(len(iqm_sim)):
        iqm_sim[j] = (iqm_sim[j] - mean)/std
    real = json.loads(df["real"].iloc[0])
    iqm_real = real["iqm"]
    for j in range(len(iqm_real)):
        iqm_real[j] = (iqm_real[j] - mean)/std        
    dic1[string] += iqm_sim
    dic2[string] += iqm_real
keys_list = list(dic1.keys())
fig, ax = plt.subplots()
res_sim = []
res_real = []
print(keys_list)
for key in keys_list:
    res_sim.append(dic1[key])
    res_real.append(dic2[key])
box1 = ax.boxplot(res_sim, positions=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], widths=0.4, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue', alpha=0.5),
                  whiskerprops=dict(color='green', linewidth=1.5),
                  capprops=dict(color='red', linewidth=1.5),
                  medianprops=dict(color='yellow', linewidth=0),
                  flierprops=dict(marker='None', color='black', markerfacecolor='red'))
box2 = ax.boxplot(res_real, positions=[1.4,2.4,3.4,4.4,5.4,6.4,7.4,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,16.4], widths=0.4, patch_artist=True,
                  boxprops=dict(facecolor='pink', color='red', alpha=0.5),
                  whiskerprops=dict(color='blue', linewidth=1.5),
                  capprops=dict(color='navy', linewidth=1.5),
                  medianprops=dict(color='green', linewidth=0),
                  flierprops=dict(marker='None', color='black', markerfacecolor='pink'))
print("Boxes:")
for box in box2['boxes']:
    path = box.get_path()  # 获取 Path 对象
    print((path.vertices[0, 1]+path.vertices[2, 1])/2,(path.vertices[2, 1]-path.vertices[0, 1])/2)  # 打印 y 坐标数据

plt.title('The Episode Lengths of RL Models with various Hyperparameter Configurations')
plt.xlabel('RL Models')
plt.ylabel('Episode Length')
plt.xticks([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2], keys_list)

# 定义自定义的图例
plt.legend([box1["boxes"][0], box2["boxes"][0]], ['SE1', 'SE2'], loc='upper right')

# 显示图表
plt.show()