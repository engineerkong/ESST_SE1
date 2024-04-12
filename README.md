conda env create -f environment.yaml
conda run -n SE1 pip install pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -extra-index-url https://download.pytorch.org/whl/cu113
conda run -n SE1 pip install -r requirements.txt
conda run -n SE1 pip install -e ./src/mygym/
conda run -n SE1 pip install -e ./src/autorl/