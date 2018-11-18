# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net_numpy import PolicyValueNetNumpy
#from policy_value_net_numpy import PolicyValueNet
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import os,sys

from backup import *
from human import *
import numpy as np




def get_equi_data( play_data,board_height,board_width):                                           
    """augment the data set by rotation and flipping                          
    play_data: [(state, mcts_prob, winner_z), ..., ...]                       
    """                                                                       
    extend_data = []                                                          
    for state, mcts_porb, winner in play_data:                                
        for i in [1, 2, 3, 4]:                                                
            # rotate counterclockwise                                         
            equi_state = np.array([np.rot90(s, i) for s in state])            
            equi_mcts_prob = np.rot90(np.flipud(                              
                mcts_porb.reshape(board_height, board_width)), i)   
            extend_data.append((equi_state,                                   
                                np.flipud(equi_mcts_prob).flatten(),          
                                winner))                                      
            # flip horizontally                                               
            equi_state = np.array([np.fliplr(s) for s in equi_state])         
            equi_mcts_prob = np.fliplr(equi_mcts_prob)                        
            extend_data.append((equi_state,                                   
                                np.flipud(equi_mcts_prob).flatten(),          
                                winner))                                      
    return extend_data                                                        



def run():
    n = 5
    #width, height = 8, 8
    width, height = 16,16
    #model_file =  'best_policy_8_8_5.model'
    model_file =  './tfData/best_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy

        #得到策略                               
        best_policy = PolicyValueNet(width, height, model_file)
        #得到策略函数
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        winner, play_data  = game.start_play(human, mcts_player, start_player=1, is_shown=1)
        play_data = list(play_data)[:]
        play_data = get_equi_data(play_data,height,width )
        backupSave(play_data,"human" )

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    curPath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curPath)
    run()
