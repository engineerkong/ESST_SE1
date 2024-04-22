import torch
import os
from stable_baselines3 import SAC

def search_files(folder_path, target_file_extension):
    target_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(target_file_extension):
                file_path = os.path.join(root, file)
                target_files.append(file_path)
    return target_files

# 加载 SB3 模型
folder_path = "../DATA-ESST/gym_sac_hps"
model_paths = search_files(folder_path,".zip")
for model_path in model_paths:
    print(model_path)
    model = SAC.load(model_path)
    parent_dir = os.path.dirname(model_path)
    # 提取 PyTorch 网络
    network = model.policy.state_dict()
    # 保存为 PyTorch 模型
    torchsave_path = "model.pth.tar"  # 指定保存路径
    torch.save(model.get_parameters(), os.path.join(parent_dir, 'model_torch.pth.tar'), _use_new_zipfile_serialization=False)
