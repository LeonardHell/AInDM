import json
from pathlib import Path
import numpy as np
import time

seed = 48
np.random.seed(seed)
n = (4,8,16,32,64)


def init_players(n, dist) -> list:
    """Initialize players with their respective win chances."""
    if dist == "uniform":
        players = np.random.uniform(0, 1, n).tolist()
        players.sort(reverse=True)
        return players
    raise ValueError(f"Unknown player skill distribution: {dist}")


rng = np.random.default_rng(42)

GROUP_MODES = ("snake", "random", "best_vs_best")
KNOCKOUT_MODES = ("best_vs_best", "random", "best_vs_worst")
SIMULATIONS_PER_CONFIGURATION = 1_000
ANALYSIS_SIMULATIONS = 1000


def play_match(player1, player2, skills):
    """Return the winner of a match using Bradley-Terry."""
    s1 = skills[player1]
    s2 = skills[player2]
    p1_wins = s1 / (s1 + s2)
    p2_wins = s2 / (s1+s2)
    if rng.random() < p1_wins:
        return player1, abs(s1 - s2)
    return player2, abs(p1_wins-p2_wins)


def create_groups(num_groups, skills, mode="snake"):
    """Distribute players into groups using the selected seeding mode."""
    n = len(skills)
    if mode not in GROUP_MODES:
        raise ValueError(f"Unknown group formation mode: {mode}")
    if not 1 <= num_groups <= n:
        raise ValueError("The number of groups must be between 1 and n.")

    groups = [[] for _ in range(num_groups)]
    ranked_players = sorted(range(n), key=lambda player: skills[player], reverse=True)

    if mode == "random":
        players = rng.permutation(n).tolist()
        for i, player in enumerate(players):
            groups[i % num_groups].append(player)
    elif mode == "best_vs_best":
        for group, players in enumerate(np.array_split(ranked_players, num_groups)):
            groups[group] = players.tolist()
    else:
        positions = list(range(num_groups)) + list(range(num_groups - 1, -1, -1))
        while len(positions) < n:
            positions.extend(positions)
        for i, player in enumerate(ranked_players):
            groups[positions[i]].append(player)

    return groups


def group_stage(groups, qualifiers_per_group, skills):
    """Play each group round-robin and return each group's qualifiers."""
    qualified = []
    sum_tension_gr = 0
    for group in groups:
        if qualifiers_per_group > len(group):
            raise ValueError("A group cannot advance more players than it contains.")

        points = {player: 0 for player in group}
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                winner, tension = play_match(group[i], group[j], skills)
                points[winner] += 1
                sum_tension_gr += tension

        ranking = sorted(
            group,
            key=lambda player: (points[player], rng.random()),
            reverse=True,
        )
        qualified.append(ranking[:qualifiers_per_group])

    return qualified, sum_tension_gr


def knockout_stage(qualified, skills, mode="best_vs_best"):
    """Play knockout rounds with the selected pairing mode."""
    if mode not in KNOCKOUT_MODES:
        raise ValueError(f"Unknown knockout pairing mode: {mode}")

    players = [player for group in qualified for player in group]
    if not players:
        raise ValueError("The knockout stage requires at least one player.")
    if len(players) & (len(players) - 1):
        raise ValueError("The number of knockout players must be a power of two.")

    sum_tension_ko = 0

    while len(players) > 1:
        if mode == "random":
            rng.shuffle(players)
            pairs = [(players[i], players[i + 1]) for i in range(0, len(players), 2)]
        else:
            players.sort(key=lambda player: skills[player], reverse=True)
            if mode == "best_vs_best":
                pairs = [(players[i], players[i + 1]) for i in range(0, len(players), 2)]
            else:
                pairs = [
                    (players[i], players[-(i + 1)])
                    for i in range(len(players) // 2)
                ]

        for player1, player2 in pairs:
            winner, tension = play_match(player1, player2, skills)
            sum_tension_ko += tension
            players.remove(player1 if winner == player2 else player2)

    return players[0], sum_tension_ko


def simulate_tournament(
    x,
    skills,
    group_mode="snake",
    qualifiers_per_group=None,
    knockout_mode="best_vs_best",
):
    """Simulate a tournament with 2**x players entering the knockout stage."""
    n = len(skills)
    if n < 1 or n & (n - 1):
        raise ValueError("The number of players must be a positive power of two.")
    if not 0 <= x <= int(np.log2(n)):
        raise ValueError(f"x must be between 0 and {int(np.log2(n))}.")

    knockout_players = 2**x
    if qualifiers_per_group is None:
        qualifiers_per_group = min(2, knockout_players)
    if (
        qualifiers_per_group < 1
        or qualifiers_per_group & (qualifiers_per_group - 1)
        or qualifiers_per_group > knockout_players
    ):
        raise ValueError(
            "qualifiers_per_group must be a power of two no greater than 2**x."
        )

    num_groups = knockout_players // qualifiers_per_group
    groups = create_groups(num_groups, skills, group_mode)
    qualified, sum_tension_gr = group_stage(groups, qualifiers_per_group, skills)
    res = (*knockout_stage(qualified, skills, knockout_mode), sum_tension_gr)
    return res


def Monte_Carlo(
    x,
    n,
    simulations=SIMULATIONS_PER_CONFIGURATION,
    group_mode="snake",
    qualifiers_per_group=None,
    knockout_mode="best_vs_best",
):
    """Estimate each player's probability of winning this tournament setup."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a positive power of two.")
    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    player_wins = [0] * n
    tension = 0
    for _ in range(simulations):
        skills = init_players(n, "uniform")
        winner, ten, ten_gr = simulate_tournament(
            x,
            skills,
            group_mode=group_mode,
            qualifiers_per_group=qualifiers_per_group,
            knockout_mode=knockout_mode,
        )
        player_wins[winner] += 1
        tension += ten + ten_gr

    return [wins / simulations for wins in player_wins] , tension / simulations


def run_analysis(n, simulations=ANALYSIS_SIMULATIONS):
    """Evaluate every valid combination of tournament settings."""
    start = time.perf_counter()
    
    results = {}

    for n in n:
        if n < 1 or n & (n - 1):
            raise ValueError("Every player count must be a positive power of two.")
        results[str(n)] = {}
        max_x = int(np.log2(n))

        for x in range(max_x + 1):
            results[str(n)][str(x)] = {}
            qualifier_counts = [
                2**power for power in range(x + 1)
            ]

            for group_mode in GROUP_MODES:
                results[str(n)][str(x)][group_mode] = {}
                for qualifiers_per_group in qualifier_counts:
                    results[str(n)][str(x)][group_mode][str(qualifiers_per_group)] = {}
                    for knockout_mode in KNOCKOUT_MODES:
                        probabilities = Monte_Carlo(
                            x,
                            n=n,
                            simulations=simulations,
                            group_mode=group_mode,
                            qualifiers_per_group=qualifiers_per_group,
                            knockout_mode=knockout_mode,
                        )
                        results[str(n)][str(x)][group_mode][str(qualifiers_per_group)][
                            knockout_mode] = probabilities
    end = time.perf_counter()
    
    runtime = end - start
    print(f"Runtime: {runtime:.4f} seconds")

    return {
        "metadata": {
            "player_counts": str(n),
            "Runtime": f"{runtime:.4f} seconds",
            "player_index": "skill rank; index 0 is the strongest",
            "simulations_per_configuration": simulations,
            "group_modes": list(GROUP_MODES),
            "qualifiers_per_group": "all powers of two up to 2**x",
            "knockout_modes": list(KNOCKOUT_MODES),
        },
        "results": results,
    }


def save_results(results):
    """Save analysis as JSON-formatted text for later loading with json."""
    output_path = Path(__file__).resolve().with_name(f"tournament_results_n_{str(n)}_{ANALYSIS_SIMULATIONS}.txt")
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return output_path


if __name__ == "__main__":
    analysis = run_analysis(n)
    results_path = save_results(analysis)
    print(f"Saved tournament analysis to {results_path}")
