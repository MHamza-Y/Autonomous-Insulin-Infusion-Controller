# Autonomous-Insulin-Infusion-Controller
Create a controller using reinforcement learning for autonomous insulin infusion.

## Installing Requirements 
It is recommended to install the requirements in virtual env.
- Enable the virtual environment (optional but recommended).
- Then run following in terminal (for gpu based requirements),
 
    `pip install -r requirements.txt`

## Starting the Training 

Run following command to start training in terminal,

`python simglucose_train.py`

or just execute `simglucose_train.py` in your ide

## Monitor Training 

Run tensorboard to monitor training using,

`tensorboard --logdir <tenorboard dir>`

Tensorboard dir is created after starting the training. Substitute the dir name in the command above.
