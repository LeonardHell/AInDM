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

        points = {player: 0 for player in group}
        # Everyone plays everyone
        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                winner = play_match(group[i], group[j])
                points[winner] += 1

        # Rank by points, then randomize ties
        ranking = sorted(
            group,
            key=lambda player: (points[player], rng.random()),
            reverse=True,
        )

        qualified.extend(ranking[:2])

    return qualified


# Knockout stage

def knockout_stage(qualified):

    players = sorted(
        qualified,
        key=lambda player: -skills[player],
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
    qualified = group_stage(groups)

    winner = knockout_stage(qualified)

    return winner

# -----------------------------
# MONTE CARLO EXPERIMENT
# -----------------------------
max_x = int(np.log2(n))
def Monte_Carlo(x,simulations = 100_000):
    
    player_wins = [0]*n
    for _ in range(simulations):
        if x==int(np.log2(n)):
            winner = knockout_stage(list(range(0, n)))
        elif x==0:
            num_groups = 1
            groups = create_groups(num_groups)
            qualified = group_stage(groups)
            winner = qualified[0]
        else:
            winner = simulate_tournament(x)
            
        player_wins[winner]+=1

    probabilities = [x/simulations for x in player_wins]
    return(probabilities)

x_prob = {}
for x in range(int(np.log2(n))):    
    x_prob[x] = Monte_Carlo(x)

print(x_prob)