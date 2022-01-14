from stable_baselines import PPO2
import glob
import json
import os
import zmq


from controller import PPOController


def main():
    list_of_files = glob.glob('./model/rl_model*.zip')
    saved_model = max(list_of_files, key=os.path.getctime)

    model = PPO2.load(saved_model)
    controller = PPOController(0, model)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:

        message = socket.recv()
        print(int(message))
        basal_rate = controller.predict(message)
        response = json.dumps(str(basal_rate)).encode('ascii')
        socket.send(response)


main()
