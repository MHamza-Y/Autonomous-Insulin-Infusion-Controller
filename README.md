# Autonomous-Insulin-Infusion-Controller
This project attempts to create a controller using reinforcement learning for autonomous insulin infusion.

Important notes,

- Training requires linux machine with cuda capable gpu.
- To test the code run the docker service. Which starts a server (returns insulin basal rate for given Blood glucose value) and client (using dexcom sandbox data) service.
- The test/train python scripts should be execute from the root folder.

## Results
Results compared with no-controller and pid-controller scenario while keeping same meal plan. Trained for 1.2 million steps with fixed hyperparameters and early stopping. The performance drops after 1.2m training steps. Which could be fixed by decreasing learning rate, cliprange etc with increasing steps to avoid catastrophic forgetting.
| Without Controller | PID Controller | PPO with LSTM Policy |
| --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/6195902/152699396-0cb972e7-f57a-42bc-bb89-1a8a88549cf2.png) | ![image](https://user-images.githubusercontent.com/6195902/152699413-c367a188-202c-4afa-aaf4-f62abbc2f03d.png) | ![image](https://user-images.githubusercontent.com/6195902/152699540-0c98d485-cf67-44c7-9035-4a3395df2c2f.png)| 


## Installing Requirements 

This project has two type of train/test scripts.
- To run stable-baselines and tensorflow-gpu==1.15.0 based scripts activate python3.7 venv and install requirements using,
  
  `pip install -r requirements-tensorflow.txt`
- To run stable-baselines3 and pytorch gpu based scripts activate python3.8 or newer venv and install requirements using
  
  `pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html`
  
To run the blood glucose controller server and sandbox simulation client install,
 - [Docker and Docker Compose](https://docs.docker.com/get-docker/) on any OS
 
    
## Starting Client/Server Services

To start Client/Server services execute the following command in the root folder,

`docker-compose up`

## Starting the Train/Test Scripts

To start the train/Test scripts run them using python while keeping root folder as working dir. Test script looks for models in /training_ws in zip format. These are automatically created after starting training.

## Monitor Training 

Run tensorboard to monitor training using,

`tensorboard --logdir <tenorboard dir>`

Tensorboard dir is created after starting the training. Substitute the dir name in the command above.

