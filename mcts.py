import math
import random
from othello import BLACK, WHITE


class MCTSNode:
    def __init__(self, state, parent=None, move=None, root_player=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.root_player = root_player

        self.children = []
        self.untried_moves = state.get_legal_moves()

        if not self.untried_moves and not state.is_game_over():
            self.untried_moves = [None]  # pass move

        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        best_score = float("-inf")
        best_children = []

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.clone()
        next_state.apply_move(move)

        child = MCTSNode(
            state=next_state,
            parent=self,
            move=move,
            root_player=self.root_player
        )

        self.children.append(child)
        return child


class MCTSBot:
    def __init__(self, simulations=1000, exploration_weight=1.4):
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def choose_move(self, game):
        root_player = game.current_player
        root = MCTSNode(
            state=game.clone(),
            root_player=root_player
        )

        for _ in range(self.simulations):
            node = root

            while (
                not node.state.is_game_over()
                and node.is_fully_expanded()
                and node.children
            ):
                node = node.best_child(self.exploration_weight)

            if not node.state.is_game_over() and not node.is_fully_expanded():
                node = node.expand()

            result = self.rollout(node.state.clone(), root_player)

            self.backpropagate(node, result)

        if not root.children:
            return None

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def rollout(self, state, root_player):
        while not state.is_game_over():
            moves = state.get_legal_moves()

            if moves:
                move = self.rollout_policy(state, moves)
            else:
                move = None

            state.apply_move(move)

        winner = state.get_winner()

        if winner == root_player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0

    def rollout_policy(self, state, moves):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_moves = [move for move in moves if move in corners]
        if corner_moves:
            return random.choice(corner_moves)

        return random.choice(moves)

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent