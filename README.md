# Autonomous-Insulin-Infusion-Controller
Create a controller using reinforcement learning for autonomous insulin infusion.

## Installing Requirements 
This project has two type of train/test scripts.
- To run stable-baselines and tensorflow-gpu==1.15.0 based scripts activate python3.7 venv and install requirements using,
  `pip install -r requirements-tensorflow.txt`
- To run stable-baselines3 and pytorch gpu based scripts activate python3.8 or newer venv and install requirements using
  `pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html`
 
    

## Starting the Train/Test Scripts

To start the train/Test scripts run them using python while keeping root folder as working dir. Test script looks for models in /training_ws in zip format. These are automatically created after starting training.

## Monitor Training 

Run tensorboard to monitor training using,

`tensorboard --logdir <tenorboard dir>`

Tensorboard dir is created after starting the training. Substitute the dir name in the command above.
