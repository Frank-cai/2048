from game2048.game import Game
from game2048.displays import Display
import torch


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    # agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=False, train=0)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from game2048.agents import MyAgent as TestAgent
    '''===================='''
    size = 4
    score_to_win = 2048

    # 59*16*12
    game = Game(size, score_to_win)
    agent = TestAgent(game, display=Display())

    scores = []
    # agent.net8.load_state_dict(torch.load("net_3_params_8.pkl", map_location='cpu'))
    agent.net7.load_state_dict(torch.load("net_3_params_7.pkl", map_location='cpu'))
    agent.net6.load_state_dict(torch.load("net_3_params_6.pkl", map_location='cpu'))
    agent.net5.load_state_dict(torch.load("net_3_params_5.pkl", map_location='cpu'))
    agent.net4.load_state_dict(torch.load("net_3_params_4.pkl", map_location='cpu'))
    # agent.net3.load_state_dict(torch.load("net_3_params_3.pkl", map_location='cpu'))
    agent.net2.load_state_dict(torch.load("net_3_params_2.pkl", map_location='cpu'))
    # agent.net1.load_state_dict(torch.load("net_3_params_1.pkl", map_location='cpu'))

    step = 0
    for i in range(N_TESTS):
        game = Game(size, score_to_win)
        agent = TestAgent(game, display=Display())

        # agent.net8.load_state_dict(torch.load("net_3_params_8.pkl", map_location='cpu'))
        agent.net7.load_state_dict(torch.load("net_3_params_7.pkl", map_location='cpu'))
        agent.net6.load_state_dict(torch.load("net_3_params_6.pkl", map_location='cpu'))
        agent.net5.load_state_dict(torch.load("net_3_params_5.pkl", map_location='cpu'))
        agent.net4.load_state_dict(torch.load("net_3_params_4.pkl", map_location='cpu'))
        # agent.net3.load_state_dict(torch.load("net_3_params_3.pkl", map_location='cpu'))
        agent.net2.load_state_dict(torch.load("net_3_params_2.pkl", map_location='cpu'))
        # agent.net1.load_state_dict(torch.load("net_3_params_1.pkl", map_location='cpu'))

        agent.play(verbose=True)
        step = step + agent.n_iter
        scores.append(game.score)

    print(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
