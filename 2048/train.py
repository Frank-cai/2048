from game2048.game import Game
from game2048.displays import Display
import torch

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    # #####################20-6 30-1
    n = 20
    b = 128
    # #####################

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

    agent.train3(BATCH_SIZE=b, NUM_EPOCHS=n)

    torch.save(agent.net_3.state_dict(), "net_3_params_8.pkl")
    agent.net_3.load_state_dict(torch.load("net_3_params_8.pkl"))

    for i in range(10):
        game = Game(size, score_to_win)
        agent = TestAgent(game, display=Display())
        agent.net_3.load_state_dict(torch.load("net_3_params_8.pkl"))
        agent.play(verbose=True)
        scores.append(game.score)

    print(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
