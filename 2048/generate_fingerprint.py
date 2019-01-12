import json
import numpy as np
from game2048.game import Game
import torch


def generate_fingerprint(AgentClass, **kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(game=game, **kwargs)

    agent.net7.load_state_dict(torch.load("net_3_params_7.pkl", map_location='cpu'))
    agent.net6.load_state_dict(torch.load("net_3_params_6.pkl", map_location='cpu'))
    agent.net5.load_state_dict(torch.load("net_3_params_5.pkl", map_location='cpu'))
    agent.net4.load_state_dict(torch.load("net_3_params_4.pkl", map_location='cpu'))
    # agent.net3.load_state_dict(torch.load("net_3_params_3.pkl", map_location='cpu'))
    agent.net2.load_state_dict(torch.load("net_3_params_2.pkl", map_location='cpu'))

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter

    '''====================
    Use your own agent here.'''
    from game2048.agents import MyAgent as TestAgent
    '''===================='''

    fingerprint = generate_fingerprint(TestAgent)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
