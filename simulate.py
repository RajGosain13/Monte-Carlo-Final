from greedybot import GreedyBot
from heuristicbot import HeuristicBot
from mcts_tdl import MCTSTDLBot
from mcts import MCTSBot
from mcts_heuristic import MCTSHeuristicBot
from mcts_combined import MCTSCombinedBot
from mcts_optimized import MCTSBotOpt
from mcts_shallow import MCTSShallowBot
from mcts_heuristic_expansion import MCTSHeuristicExpansionBot
import time
import os
from othello import Othello, BLACK, WHITE

def play_game(bot_black, bot_white):
    game = Othello()

    while not game.is_game_over():
        player = game.current_player
        bot = bot_black if player == BLACK else bot_white

        legal_moves = game.get_legal_moves(player)

        if not legal_moves:
            game.apply_move(None, player)
            continue

        move = bot.choose_move(game.clone(), player)

        if move not in legal_moves:
            raise ValueError(f"Bot made illegal move {move} for player {player}")

        game.apply_move(move, player)

    winner = game.get_winner()
    score = game.get_score()

    return winner, score

def evaluate_matchup(bot_a, bot_b, n_games=1000):
    a_wins = 0
    b_wins = 0
    draws = 0

    total_a_discs = 0
    total_b_discs = 0

    for i in range(n_games):
        if i % 2 == 0:
            # bot_a is black
            winner, score = play_game(bot_a, bot_b)

            a_score = score["black"]
            b_score = score["white"]

            if winner == BLACK:
                a_wins += 1
            elif winner == WHITE:
                b_wins += 1
            else:
                draws += 1

        else:
            # bot_a is white
            winner, score = play_game(bot_b, bot_a)

            a_score = score["white"]
            b_score = score["black"]

            if winner == WHITE:
                a_wins += 1
            elif winner == BLACK:
                b_wins += 1
            else:
                draws += 1

        total_a_discs += a_score
        total_b_discs += b_score

    return {
        "games": n_games,
        "bot_a_wins": a_wins,
        "bot_b_wins": b_wins,
        "draws": draws,
        "bot_a_winrate": a_wins / n_games,
        "bot_b_winrate": b_wins / n_games,
        "draw_rate": draws / n_games,
        "avg_bot_a_discs": total_a_discs / n_games,
        "avg_bot_b_discs": total_b_discs / n_games,
        "avg_disc_margin_for_a": (total_a_discs - total_b_discs) / n_games,
    }

mcts_bot = MCTSBot()
greedy_bot = GreedyBot()
heuristic_bot = HeuristicBot()
mctstd_bot = MCTSTDLBot()
mcts_heuristic_bot = MCTSHeuristicBot()
mcts_combined_bot = MCTSCombinedBot()
mcts_optimized_bot = MCTSBotOpt()
mcts_shallow_bot = MCTSShallowBot()
mcts_expansion_bot = MCTSHeuristicExpansionBot()

td1_bot = MCTSTDLBot()
if not os.path.exists("tdl_weights.txt"):
    td1_bot.train(num_games=10000)
    print(f"Trained weight: {td1_bot.w}")
else:
    print(f"Loaded weight: {td1_bot.w}")

print("mcts vs mcts_combined")
start = time.time()
results = evaluate_matchup(mcts_bot, mcts_combined_bot, n_games=500)
elapsed = time.time() - start
print(results)
print(f"Time: {elapsed:.2f}s")

"""
game = Othello()
n = 10

bots = {
    "GreedyBot":     GreedyBot(),
    "HeuristicBot":  HeuristicBot(),
    "MCTSBot":       MCTSBot(),
    "MCTSTDLBot":    MCTSTDLBot(),
    "MCTSHeuristic": MCTSHeuristicBot(),
    "MCTSCombined":  MCTSCombinedBot(),
    "MCTSOptimized": MCTSBotOpt(),
    "MCTSShallow":   MCTSShallowBot(),
}

for name, bot in bots.items():
    total = 0
    for _ in range(n):
        start = time.time()
        bot.choose_move(game.clone(), game.current_player)
        total += time.time() - start
    print(f"{name}: {total/n:.4f}s avg over {n} runs")
"""
