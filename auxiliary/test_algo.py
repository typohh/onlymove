import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import beta

class AdaptiveRanker:
    """
    An adaptive ranking system that uses a Beta distribution to model win rate.
    The learning rate is modulated by the normalized likelihood of the target winrate.
    """

    def __init__(self, initial_rank=1000, target_winrate=0.5):
        self.rank = initial_rank
        self.target_winrate = target_winrate
        self.alpha_param = 2
        self.beta_param = 2
        self.epsilon = 0.01
        self.k = 1

    def update_rank(self, is_correct):
        win_rate_estimate = self.alpha_param / (self.alpha_param + self.beta_param)

        # Calculate the value of the Beta PDF at our target winrate
        pdf_val_at_target = beta.pdf(self.target_winrate, self.alpha_param, self.beta_param)

        # Find the peak of the distribution (the mode) to get the max PDF value.
        mode = (self.alpha_param - 1) / (self.alpha_param + self.beta_param - 2)

        # The maximum value of the PDF is the value at its mode.
        pdf_max = beta.pdf(mode, self.alpha_param, self.beta_param)

        prob = pdf_val_at_target / pdf_max
        # --------------------------------------------------------------------------

        if is_correct:
            self.alpha_param += 1
        else:
            self.beta_param += 1

        # The rest of the logic remains the same
        alpha_lr = 0.5 + prob / 2 - self.epsilon
        # alpha_lr = self.epsilon + prob * (1 - 2 * self.epsilon)

        self.alpha_param = ( 2 + self.alpha_param - 2 ) * alpha_lr - 2
        self.beta_param = ( 2 + self.beta_param - 2 ) * alpha_lr - 2

        alpha_momentum = self.epsilon + prob * (1 - 2 * self.epsilon)
        is_aligned = (is_correct == (win_rate_estimate > self.target_winrate))
        momentum = 1 / alpha_momentum if is_aligned else alpha_momentum

        direction = 1 if is_correct else -1
        self.rank += self.k * momentum * direction

def run_multiple_simulations(num_simulations):
    all_rank_histories = []
    initial_rank = 1000
    true_strength_1, true_strength_2 = 1800, 1400
    num_games_total, switch_point = 1500, 300
    target_winrate = 0.5
    rating_offset = 400 * math.log10(1 / target_winrate - 1)
    target_rank_1, target_rank_2 = true_strength_1 + rating_offset, true_strength_2 + rating_offset
    true_strength_history = [true_strength_1 if i < switch_point else true_strength_2 for i in range(num_games_total)]
    target_rank_history = [target_rank_1 if i < switch_point else target_rank_2 for i in range(num_games_total)]

    for _ in range(num_simulations):
        ranker = AdaptiveRanker(initial_rank=initial_rank, target_winrate=target_winrate)
        rank_history = []
        for game_num in range(num_games_total):
            rank_history.append(ranker.rank)
            true_strength = true_strength_history[game_num]
            win_prob = 1 / (1 + 10 ** ((ranker.rank - true_strength) / 400))
            is_correct = np.random.rand() < win_prob
            ranker.update_rank(is_correct)
        all_rank_histories.append(rank_history)

    return all_rank_histories, true_strength_history, target_rank_history


# --- Run Simulation and Plot ---
num_runs = 5
all_histories, true_strength_hist, target_rank_hist = run_multiple_simulations(num_simulations=num_runs)

plt.figure(figsize=(14, 8))

for i, history in enumerate(all_histories):
    plt.plot(history, color='blue', alpha=0.35)

plt.plot(true_strength_hist, label="True Strength (50% Win Rate)", color='red', linestyle='--', linewidth=2.5, zorder=5)
plt.plot(target_rank_hist, label="Target Rating (80% Win Rate)", color='green', linestyle=':', linewidth=3, zorder=5)
plt.axvline(x=300, color='gray', linestyle='-.', label='True Strength Change', zorder=4)

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2.5, linestyle='--'),
                Line2D([0], [0], color='green', lw=3, linestyle=':'),
                Line2D([0], [0], color='gray', lw=1, linestyle='-.')]

plt.title(f"{num_runs} Runs", fontsize=16)
plt.xlabel("Game Number", fontsize=12)
plt.ylabel("Rating", fontsize=12)
plt.legend(custom_lines, ['Adaptive Rating Runs', 'True Strength (50% Win Rate)', 'Target Rating (80% Win Rate)', 'True Strength Change'], fontsize=12)
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.show()
