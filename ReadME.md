# Adversarial Search Suite

This repository contains Python implementations of four fundamental adversarial search algorithms, tested against a standard Tic-Tac-Toe environment. 

## Algorithms Implemented

1. **Minimax Search:** A recursive algorithm that explores the entire game tree to find the optimal move, assuming the opponent plays perfectly.
2. **Alpha-Beta Pruning:** An optimization of Minimax that prunes branches of the game tree that cannot possibly influence the final decision, significantly reducing computation time.
3. **Heuristic Alpha-Beta Search:** A depth-limited version of Alpha-Beta that stops at a predetermined depth limit and evaluates the non-terminal state using a heuristic evaluation function.
4. **Monte-Carlo Tree Search (MCTS):** A heuristic search algorithm that uses random sampling (rollouts) to evaluate states and builds an asymmetric tree. It balances exploration and exploitation using the UCB1 formula:
   
   $$UCT = \frac{w_i}{n_i} + c \sqrt{\frac{\ln(N)}{n_i}}$$

## Project Structure
* `game.py`: Contains the state representation and rules for the environment (Tic-Tac-Toe).
* `algorithms.py`: Contains the logic for Minimax, Alpha-Beta, Heuristic Alpha-Beta, and MCTS.
* `test_algorithms.py`: A `unittest` suite verifying the correctness of each algorithm under specific board states.

## How to Run the Tests
Ensure you have Python 3 installed. Navigate to the project directory and run:
```bash
python -m unittest test_algorithms.py

---

### **2. `game.py` (The Environment)**

This file contains the logic for Tic-Tac-Toe, which is necessary for the algorithms to traverse states, evaluate moves, and check for terminal conditions.

```python
# game.py
import copy

class TicTacToe:
    """
    A simple Tic-Tac-Toe environment.
    Players: 1 (Maximizer/X) and -1 (Minimizer/O). 0 represents an empty space.
    The board is a 1D list of 9 elements.
    """
    def __init__(self, board=None, current_player=1):
        self.board = board if board else [0] * 9
        self.current_player = current_player

    def get_legal_moves(self):
        """Returns a list of available indices (0-8)."""
        return [i for i, spot in enumerate(self.board) if spot == 0]

    def apply_move(self, move):
        """Returns a new game state with the move applied."""
        new_board = copy.deepcopy(self.board)
        new_board[move] = self.current_player
        return TicTacToe(new_board, -self.current_player)

    def is_terminal(self):
        """Checks if the game has ended (win or draw)."""
        return self.get_winner() is not None or len(self.get_legal_moves()) == 0

    def get_winner(self):
        """Returns 1 if X wins, -1 if O wins, or None if no winner yet."""
        winning_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # Cols
            (0, 4, 8), (2, 4, 6)             # Diagonals
        ]
        for a, b, c in winning_positions:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def evaluate_heuristic(self):
        """
        A simple heuristic for depth-limited search.
        +10 for a win, -10 for a loss. 
        Otherwise, scores based on center control to break ties.
        """
        winner = self.get_winner()
        if winner == 1: return 10
        if winner == -1: return -10
        
        score = 0
        if self.board[4] == 1: score += 1
        elif self.board[4] == -1: score -= 1
        return score