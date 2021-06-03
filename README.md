# STROKE PREDICTION

ML tool to predict risk of having stroke.

Stage 0: inital notebook and version of packages

To start training, run:
```bash
python train.py
```

Stage 1: 
- Split the initial notebooks into /src/config.py, /src/get-data.py, /src/trainer.py, /src/model-train.py, /src/dataset.py , /src/utils.py, and train.py in the following basic structure
```bash
├──src
│   ├── dataset.py
│   ├── model_train.py
│   └── get-data.py
│   ├── config.py
│   ├── trainer.py
│   └── utils.py
├── train.py
	
```
- Add requirements.txt for better installation.
**Note**: check commits in the branch to see the code progression through time.

## Install 
Create virtual environment
```bash
conda create -n muenv python=3.7
conda activate myenv
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download and set up data by running
```bash
bash setup-data.sh
```

## Usage
Run
```bash
python train.py
```


