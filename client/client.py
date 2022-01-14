# This client is just for testing server with simulation data
#
#
import json

import zmq

from BGScheduler import get_current_bg
import schedule


context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://insulin-cont-server:5555")


def get_basal_insulin():
    current_bg = json.dumps(get_current_bg()).encode('ascii')
    print(f'Current BG:{current_bg}')
    socket.send(current_bg)
    response = socket.recv()
    print(response)
    return response




def main():
    basal_rate = get_basal_insulin()
    schedule.every(1).seconds.do(get_basal_insulin)

    while True:
        schedule.run_pending()


if __name__ == '__main__':
    main()
