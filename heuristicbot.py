POSITION_WEIGHTS = [
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,  -2, -1, -1, -1, -1,  -2,  10],
    [5,   -2, -1, -1, -1, -1,  -2,   5],
    [5,   -2, -1, -1, -1, -1,  -2,   5],
    [10,  -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
]

class HeuristicBot:
    def choose_move(self, board, player):
        moves = board.get_legal_moves(player)
        if not moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in moves:
            new_board = board.clone()
            new_board.apply_move(move, player)

            score = self.evaluate(new_board, player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate(self, board, player):
        score = 0

        for r in range(8):
            for c in range(8):
                if board.board[r][c] == player:
                    score += POSITION_WEIGHTS[r][c]
                elif board.board[r][c] == -player:
                    score -= POSITION_WEIGHTS[r][c]

        my_moves = len(board.get_legal_moves(player))
        opp_moves = len(board.get_legal_moves(-player))
        score += 5 * (my_moves - opp_moves)

        return score