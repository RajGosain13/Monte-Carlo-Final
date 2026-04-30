from othello import Othello

game = Othello()

while not game.is_game_over():
    game.print_board()
    print("Current player:", "BLACK" if game.current_player == 1 else "WHITE")

    moves = game.get_legal_moves()

    if not moves:
        print("No moves available. Passing.")
        game.apply_move(None)
        continue

    print("Legal moves:", moves)
    move = eval(input("Enter move (row, col): "))

    game.apply_move(move)

game.print_board()
print("Game over!")
print("Winner:", game.get_winner())
print("Score:", game.get_score())