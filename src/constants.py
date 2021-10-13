from src.envs.windy_gridworld import WindyGridworldEnv
from src.envs.blackjack import BlackjackEnv
from src.envs.gridworld import GridworldEnv
from src.sarsa import sarsa, expected_sarsa

NAME2ENV = {
    "windygridworld": WindyGridworldEnv,
    "blackjack": BlackjackEnv,
    "gridworld": GridworldEnv,
}

NAME2ALG = {"expected_sarsa": expected_sarsa, "sarsa": sarsa}
