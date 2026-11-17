import numpy as np

seed = 47
np.random.seed(seed)

def init_players(n, dist) -> list:
    """
    Initializes a list of players with their respective win chances.
    n has to be a 
    """
    if dist == "uniform":
        players = np.random.uniform(0, 1, n).tolist()
        players.sort(reverse=True)
        return players

