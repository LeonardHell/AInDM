import numpy as np

seed = 47
np.random.seed(seed)
n=16
def init_players(n, dist) -> list:
    """
    Initializes a list of players with their respective win chances.
    n has to be a 
    """
    if dist == "uniform":
        players = np.random.uniform(0, 1, n).tolist()
        players.sort(reverse=True)
        return players
    
skills = init_players(n, "uniform")


rng = np.random.default_rng(42)

# Define winer of a game 

def play_match(player1, player2):
    """Return the winner of a match using Bradley-Terry."""
    
    s1 = skills[player1]
    s2 = skills[player2]

    p1_wins = s1 / (s1 + s2)

    if rng.random() < p1_wins:
        return player1
    else:
        return player2


#Allocation to groups using snake seeding

def create_groups(num_groups):
    """Distribute players in groupsusing snake seeding."""
    groups = [[] for _ in range(num_groups)]
    x = num_groups
    positions = list(range(0, x)) + list(range(x-1, -1, -1))
    while len(positions) < len(skills):
        positions += positions
    for i,skill in enumerate(skills):
        
        groups[positions[i]].append(i)

    return groups


# Results of group phase

def group_stage(groups):
    """Each player plays every other player.
       The top two from each group qualify.
    """
    
    qualified = []

    for group in groups:

        #points = {player: 0 for player in group}
        points = [0] * len(group)
        # Everyone plays everyone
        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                winner = play_match(group[i], group[j])
                points[winner] += 1

        # Rank by points, then randomize ties
        ranking = sorted(
            group,
            key=lambda p: (points[p], rng.random()),
            reverse=True
        )

        qualified.append(ranking[:2])

    return qualified


# Knockout stage

def knockout_stage(qualified):

    players = sorted(
        qualified,
        key=lambda p: -skills[p]
    )

    while len(players) > 1:

        winners = []

        # Best against best
        for i in range(0, len(players), 2):

            winner = play_match(players[i], players[i + 1])
            winners.append(winner)

        players = winners

    return players[0]


def simulate_tournament(x):
    num_groups = 2 ** (x - 1)
    groups = create_groups(num_groups)
    print(groups)
    qualified = group_stage(groups)

    winner = knockout_stage(qualified)

    return winner

print(simulate_tournament(2))
# -----------------------------
# MONTE CARLO EXPERIMENT
# -----------------------------
max_x = int(np.log2(n))
def Monte_Carlo(x,simulations = 100000):
    player_wins = [0]*n
    print("a")
    print(simulate_tournament(x))
    for _ in range(simulations):
        
        player_wins[simulate_tournament(x)]+=1


    probabilities = [x/simulations for x in player_wins]
    return(probabilities)
print(Monte_Carlo(2))

# -----------------------------
# 8. OPTIMAL NUMBER OF ROUNDS
# -----------------------------

best_x = x_values[np.argmax(probabilities)]

print("\nOptimal number of knockout rounds:", best_x)


# -----------------------------
# 9. VISUALIZATION
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    x_values,
    probabilities,
    marker="o"
)

plt.xlabel("Number of knockout rounds (x)")
plt.ylabel("Probability strongest player wins")
plt.title("Monte Carlo Tournament Optimization")

plt.xticks(x_values)
plt.grid(True)
plt.show()
