# test_algorithms.py
import unittest
import math
from game import TicTacToe
from algorithms import minimax, alpha_beta_search, heuristic_alpha_beta, mcts

class TestSearchAlgorithms(unittest.TestCase):

    def setUp(self):
        # 1 = X (Maximizer), -1 = O (Minimizer), 0 = Empty
        # X is one move away from winning (index 2)
        # O X 0
        # O X 0
        # 0 0 0
        self.win_in_one_board = [
            -1,  1,  0,
            -1,  1,  0,
             0,  0,  0
        ]
        
        # O is about to win, X must block at index 6
        # O 0 0
        # X X O
        # 0 0 O
        self.block_board = [
            -1,  0,  0,
             1,  1, -1,
             0,  0, -1
        ]

    def test_minimax(self):
        game = TicTacToe(self.win_in_one_board, current_player=1)
        score, move = minimax(game, depth=0, is_maximizing=True)
        self.assertEqual(move, 2, "Minimax failed to find the winning move.")
        self.assertEqual(score, 10, "Minimax failed to calculate the winning score.")

    def test_alpha_beta(self):
        game = TicTacToe(self.win_in_one_board, current_player=1)
        score, move = alpha_beta_search(game, depth=0, alpha=-math.inf, beta=math.inf, is_maximizing=True)
        self.assertEqual(move, 2, "Alpha-Beta failed to find the winning move.")
        self.assertEqual(score, 10, "Alpha-Beta failed to calculate the winning score.")

    def test_alpha_beta_block(self):
        game = TicTacToe(self.block_board, current_player=1)
        score, move = alpha_beta_search(game, depth=0, alpha=-math.inf, beta=math.inf, is_maximizing=True)
        self.assertEqual(move, 6, "Alpha-Beta failed to block the opponent.")

    def test_heuristic_alpha_beta(self):
        game = TicTacToe(self.win_in_one_board, current_player=1)
        # Restrict depth limit drastically to test if the heuristic kicks in.
        # Even with depth 1, it should see the immediate win and return 10.
        score, move = heuristic_alpha_beta(game, depth=0, depth_limit=1, alpha=-math.inf, beta=math.inf, is_maximizing=True)
        self.assertEqual(move, 2, "Heuristic AB failed to find the immediate win.")

    def test_mcts(self):
        game = TicTacToe(self.win_in_one_board, current_player=1)
        # MCTS is probabilistic, but with 1000 iterations on a 1-move-to-win board, 
        # it should find the win reliably.
        move = mcts(game, iterations=1000)
        self.assertEqual(move, 2, "MCTS failed to find the clear winning move.")

if __name__ == '__main__':
    unittest.main()