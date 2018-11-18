from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import os,sys

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p
    #人工步骤
    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)



n = 5
width, height = 8, 8
model_file =  'best_policy_8_8_5.model'

board = Board(width=width, height=height, n_in_row=n)
game = Game(board)

# ############### human VS AI ###################
# load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

# best_policy = PolicyValueNet(width, height, model_file = model_file)
# mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

# load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
try:
    policy_param = pickle.load(open(model_file, 'rb'))
except:
    policy_param = pickle.load(open(model_file, 'rb'),
                               encoding='bytes')  # To support python3
#得到策略                               
best_policy = PolicyValueNetNumpy(width, height, policy_param)
#得到策略函数
mcts_player1 = MCTSPlayer(best_policy.policy_value_fn,
                         c_puct=5,
                         n_playout=400)  # set larger n_playout for better performance

mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                         c_puct=5,
                         n_playout=400)  

# uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
# mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

# human player, input your move in the format: 2,3
human = Human()

player1 =  mcts_player1
player2 =  mcts_player2
start_player=0 
is_shown=1

game.board.init_board(start_player)
p1, p2 = game.board.players
player1.set_player_ind(p1)
player2.set_player_ind(p2)
players = {p1: player1, p2: player2}

