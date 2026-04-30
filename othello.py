EMPTY = 0
BLACK = 1
WHITE = -1

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class Othello:
    def __init__(self):
        # 8x8 board using lists
        self.board = [[EMPTY for _ in range(8)] for _ in range(8)]

        # initial setup
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE

        self.current_player = BLACK

    def clone(self):
        new_game = Othello()
        new_game.board = [row[:] for row in self.board]  # deep copy
        new_game.current_player = self.current_player
        return new_game

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def get_legal_moves(self, player=None):
        if player is None:
            player = self.current_player

        moves = []

        for r in range(8):
            for c in range(8):
                if self.board[r][c] != EMPTY:
                    continue

                if self._is_valid_move(r, c, player):
                    moves.append((r, c))

        return moves

    def _is_valid_move(self, r, c, player):
        opponent = -player

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            found_opponent = False

            while self.in_bounds(nr, nc) and self.board[nr][nc] == opponent:
                found_opponent = True
                nr += dr
                nc += dc

            if found_opponent and self.in_bounds(nr, nc) and self.board[nr][nc] == player:
                return True

        return False

    def apply_move(self, move, player=None):
        if player is None:
            player = self.current_player

        if move is None:
            # pass move
            self.current_player = -player
            return

        r, c = move

        if not self._is_valid_move(r, c, player):
            raise ValueError(f"Invalid move: {move}")

        self.board[r][c] = player
        self._flip_discs(r, c, player)

        self.current_player = -player

    def _flip_discs(self, r, c, player):
        opponent = -player

        for dr, dc in DIRECTIONS:
            discs_to_flip = []
            nr, nc = r + dr, c + dc

            while self.in_bounds(nr, nc) and self.board[nr][nc] == opponent:
                discs_to_flip.append((nr, nc))
                nr += dr
                nc += dc

            if self.in_bounds(nr, nc) and self.board[nr][nc] == player:
                for rr, cc in discs_to_flip:
                    self.board[rr][cc] = player

    def has_any_moves(self, player):
        return len(self.get_legal_moves(player)) > 0

    def is_game_over(self):
        return not self.has_any_moves(BLACK) and not self.has_any_moves(WHITE)

    def get_winner(self):
        black_count = sum(cell == BLACK for row in self.board for cell in row)
        white_count = sum(cell == WHITE for row in self.board for cell in row)

        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return 0  # draw

    def print_board(self):
        symbols = {BLACK: "B", WHITE: "W", EMPTY: "."}

        print("  0 1 2 3 4 5 6 7")
        for i in range(8):
            row = " ".join(symbols[self.board[i][j]] for j in range(8))
            print(f"{i} {row}")

    def get_score(self):
        return {
            "black": sum(cell == BLACK for row in self.board for cell in row),
            "white": sum(cell == WHITE for row in self.board for cell in row),
        }