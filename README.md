# Autonomous-Insulin-Infusion-Controller
This project attempts to create a controller using reinforcement learning for autonomous insulin infusion.

Important notes,

- Training requires linux machine with cuda capable gpu.
- To test the code run the docker service. Which starts a server (returns insulin basal rate for given Blood glucose value) and client (using dexcom sandbox data) service.
- The test/train python scripts should be execute from the root folder.




## Installing Requirements 

This project has two type of train/test scripts.
- To run stable-baselines and tensorflow-gpu==1.15.0 based scripts activate python3.7 venv and install requirements using,
  
  `pip install -r requirements-tensorflow.txt`
- To run stable-baselines3 and pytorch gpu based scripts activate python3.8 or newer venv and install requirements using
  
  `pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html`
  
To run the blood glucose controller server and sandbox simulation client install,
 - [Docker and Docker Compose](https://docs.docker.com/get-docker/) on any OS
 
    
## Starting Sandbox Data Simulation Client/Server

To start Client/Server services execute the following command in the root folder,

`docker-compose up`

## Starting the Train/Test Scripts

To start the train/Test scripts run them using python while keeping root folder as working dir. Test script looks for models in /training_ws in zip format. These are automatically created after starting training.

## Monitor Training 

Run tensorboard to monitor training using,

`tensorboard --logdir <tenorboard dir>`

Tensorboard dir is created after starting the training. Substitute the dir name in the command above.

