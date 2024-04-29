### Build dependencies

conda env create -f environment.yaml
conda run -n SE1 pip install pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -extra-index-url https://download.pytorch.org/whl/cu113
conda run -n SE1 pip install -r requirements.txt
conda run -n SE1 pip install -e ./src/mygym/
conda run -n SE1 pip install -e ./src/autorl/

### Run experiment 1 (Algorithms and Reward Functions)

# train
python ./src/mygym/myGym/train_test.py task=gym_sac 
# test on SE1
python ./src/mygym/myGym/train_test.py task=gym_sac test=1 
# test on SE2
roslaunch niryo_robot_bringup desktop_gazebo_simulation.launch
rosrun niryo_robot_bringup deploy.py 
# evaluate
python boxplot_el_ex1.py
python boxplot_nr_ex1.py

### Run experiment 2 (Hperparameters)

# train using AutoRL schedule
phases run combo=sac_mygym ls=sac slurm=local phases=10k task=mygym_sac num_confs=16 num_seeds=3 wandb.entity=entity_name wandb.project=project_name wandb.experiment_tag=experiment_tag 
# test on SE1
python ./src/mygym/myGym/train_test.py task=gym_sac test=1 
# test on SE2
roslaunch niryo_robot_bringup desktop_gazebo_simulation.launch
rosrun niryo_robot_bringup deploy.py 
# evaluate
python boxplot_el_ex2.py
python boxplot_nr_ex2.py