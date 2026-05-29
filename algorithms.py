# algorithms.py
import math
import random

# ==========================================
# 1. Minimax Search
# ==========================================
def minimax(game_state, depth, is_maximizing):
    """
    Standard Minimax algorithm without pruning.
    Returns: (best_score, best_move)
    """
    if game_state.is_terminal():
        winner = game_state.get_winner()
        score = 10 if winner == 1 else -10 if winner == -1 else 0
        return score, None

    best_move = None
    if is_maximizing:
        max_eval = -math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = minimax(next_state, depth + 1, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = minimax(next_state, depth + 1, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move

# ==========================================
# 2. Alpha-Beta Pruning
# ==========================================
def alpha_beta_search(game_state, depth, alpha, beta, is_maximizing):
    """
    Minimax enhanced with Alpha-Beta pruning to reduce branch traversal.
    Returns: (best_score, best_move)
    """
    if game_state.is_terminal():
        winner = game_state.get_winner()
        score = 10 if winner == 1 else -10 if winner == -1 else 0
        return score, None

    best_move = None
    if is_maximizing:
        max_eval = -math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = alpha_beta_search(next_state, depth + 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break # Beta cutoff
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = alpha_beta_search(next_state, depth + 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break # Alpha cutoff
        return min_eval, best_move

# ==========================================
# 3. Heuristic Alpha-Beta Search
# ==========================================
def heuristic_alpha_beta(game_state, depth, depth_limit, alpha, beta, is_maximizing):
    """
    Depth-limited Alpha-Beta search that falls back on a heuristic evaluation 
    function when the depth limit is reached before a terminal state.
    Returns: (best_score, best_move)
    """
    if game_state.is_terminal():
        winner = game_state.get_winner()
        score = 10 if winner == 1 else -10 if winner == -1 else 0
        return score, None
        
    # Cutoff test
    if depth == depth_limit:
        return game_state.evaluate_heuristic(), None

    best_move = None
    if is_maximizing:
        max_eval = -math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = heuristic_alpha_beta(next_state, depth + 1, depth_limit, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in game_state.get_legal_moves():
            next_state = game_state.apply_move(move)
            eval_score, _ = heuristic_alpha_beta(next_state, depth + 1, depth_limit, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return min_eval, best_move

# ==========================================
# 4. Monte-Carlo Tree Search (MCTS)
# ==========================================
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = state.get_legal_moves()

    def uct_select_child(self, exploration_weight=1.41):
        """Uses the UCB1 formula to select the best child node."""
        s = max(self.children, key=lambda c: (c.wins / c.visits) + 
                exploration_weight * math.sqrt(math.log(self.visits) / c.visits))
        return s

def mcts(root_state, iterations=1000):
    """
    Executes the 4 phases of MCTS: Selection, Expansion, Simulation, and Backpropagation.
    Returns: best_move
    """
    root_node = MCTSNode(root_state)

    for _ in range(iterations):
        node = root_node
        
        # 1. Selection
        while not node.untried_moves and node.children:
            node = node.uct_select_child()

        # 2. Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            next_state = node.state.apply_move(move)
            child_node = MCTSNode(next_state, parent=node, move=move)
            node.children.append(child_node)
            node = child_node

        # 3. Simulation (Rollout)
        sim_state = node.state
        while not sim_state.is_terminal():
            possible_moves = sim_state.get_legal_moves()
            sim_state = sim_state.apply_move(random.choice(possible_moves))

        # 4. Backpropagation
        winner = sim_state.get_winner()
        while node is not None:
            node.visits += 1
            # Check if the player who made the move leading to this node won
            player_who_just_moved = -node.state.current_player 
            if winner == player_who_just_moved:
                node.wins += 1
            elif winner is None: # Draw is slightly rewarded
                node.wins += 0.5 
            node = node.parent

    # Return the move of the most visited child
    return max(root_node.children, key=lambda c: c.visits).move