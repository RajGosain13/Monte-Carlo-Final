from othello import Othello, BLACK, WHITE
from mcts import MCTSBot


def player_name(player):
    return "BLACK" if player == BLACK else "WHITE"


def get_human_move(moves):
    print("Legal moves:", moves)

    while True:
        raw = input("Enter move as row,col: ")

        try:
            row_str, col_str = raw.split(",")
            move = (int(row_str.strip()), int(col_str.strip()))
        except ValueError:
            print("Invalid format. Example: 2,3")
            continue

        if move in moves:
            return move

        print("Invalid move. Please try again.")


def choose_game_mode():
    print("Choose game mode:")
    print("1. Human vs Human")
    print("2. Human BLACK vs MCTS WHITE")
    print("3. MCTS BLACK vs Human WHITE")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice in {"1", "2", "3"}:
            return choice

        print("Invalid choice.")


game = Othello()
mode = choose_game_mode()

bot = MCTSBot(simulations=1000)

if mode == "1":
    bot_player = None
elif mode == "2":
    bot_player = WHITE
else:
    bot_player = BLACK


while not game.is_game_over():
    game.print_board()
    print("Current player:", player_name(game.current_player))

    moves = game.get_legal_moves()

    if not moves:
        print("No moves available. Passing.")
        game.apply_move(None)
        continue

    if game.current_player == bot_player:
        move = bot.choose_move(game)
        print("Bot chose:", move)
    else:
        move = get_human_move(moves)

    game.apply_move(move)


game.print_board()
print("Game over!")

winner = game.get_winner()

if winner == BLACK:
    print("Winner: BLACK")
elif winner == WHITE:
    print("Winner: WHITE")
else:
    print("Winner: Draw")

print("Score:", game.get_score())