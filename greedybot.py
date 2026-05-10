class GreedyBot:
    def choose_move(self, game, player):
        moves = game.get_legal_moves(player)

        if not moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in moves:
            test_game = game.clone()
            test_game.apply_move(move, player)

            score = sum(cell == player for row in test_game.board for cell in row)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move