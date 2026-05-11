import math
import random
from othello import BLACK, WHITE, Othello


SQUARE_WEIGHTS = [
    [ 100, -20,  10,  5,  5,  10, -20,  100],
    [ -20, -50,  -2, -2, -2,  -2, -50,  -20],
    [  10,  -2,   5,  1,  1,   5,  -2,   10],
    [   5,  -2,   1,  0,  0,   1,  -2,    5],
    [   5,  -2,   1,  0,  0,   1,  -2,    5],
    [  10,  -2,   5,  1,  1,   5,  -2,   10],
    [ -20, -50,  -2, -2, -2,  -2, -50,  -20],
    [ 100, -20,  10,  5,  5,  10, -20,  100],
]

MAX_SCORE = sum(abs(SQUARE_WEIGHTS[r][c]) for r in range(8) for c in range(8))


def _board_score(state, player):
    score = 0
    for r in range(8):
        for c in range(8):
            if state.board[r][c] == player:
                score += SQUARE_WEIGHTS[r][c]
            elif state.board[r][c] == -player:
                score -= SQUARE_WEIGHTS[r][c]
    return score / MAX_SCORE


def _sigmoid(x):
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _predict(state, player, w):
    return _sigmoid(w * _board_score(state, player))


class MCTSNode:
    def __init__(self, state, parent=None, move=None, root_player=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.root_player = root_player

        self.children = []

        moves = state.get_legal_moves()
        if not moves and not state.is_game_over():
            moves = [None]
        self.untried_moves = moves

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
        best_i = max(
            range(len(self.untried_moves)),
            key=lambda i: SQUARE_WEIGHTS[self.untried_moves[i][0]][self.untried_moves[i][1]]
            if self.untried_moves[i] is not None else 0
        )
        move = self.untried_moves.pop(best_i)

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


class MCTSCombinedBot:
    def __init__(self, simulations=100, exploration_weight=1.4, alpha=0.1, epsilon=0.1):
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.alpha = alpha
        self.epsilon = epsilon
        self.w = 0.0

        try:
            with open("tdl_weights.txt", "r") as f:
                self.w = float(f.read())
        except FileNotFoundError:
            pass

    def choose_move(self, game, player=None):
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
        corner_moves = [m for m in moves if m in corners]
        if corner_moves:
            return random.choice(corner_moves)

        if random.random() < 0.5:
            player = state.current_player
            best_move = None
            best_v = -1.0
            for move in moves:
                next_s = state.clone()
                next_s.apply_move(move)
                v = _predict(next_s, player, self.w)
                if v > best_v:
                    best_v = v
                    best_move = move
            return best_move

        return random.choice(moves)

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def train(self, num_games=3000):
        for _ in range(num_games):
            self._play_and_learn()

        with open("tdl_weights.txt", "w") as f:
            f.write(str(self.w))

    def _play_and_learn(self):
        state = Othello()
        history = []

        while not state.is_game_over():
            player = state.current_player
            history.append((state.clone(), player))

            moves = state.get_legal_moves()
            if not moves:
                state.apply_move(None)
                continue

            if random.random() < self.epsilon:
                move = random.choice(moves)
            else:
                move = self._greedy_move(state)

            state.apply_move(move)

        winner = state.get_winner()

        for visited_state, player in history:
            result = 1.0 if winner == player else (0.5 if winner == 0 else 0.0)
            v = _predict(visited_state, player, self.w)
            f = _board_score(visited_state, player)
            self.w += self.alpha * (result - v) * v * (1.0 - v) * f

    def _greedy_move(self, state):
        moves = state.get_legal_moves()
        if not moves:
            return None

        player = state.current_player
        best_v, best_move = -1.0, None

        for move in moves:
            next_s = state.clone()
            next_s.apply_move(move)
            v = _predict(next_s, player, self.w)
            if v > best_v:
                best_v, best_move = v, move

        return best_move