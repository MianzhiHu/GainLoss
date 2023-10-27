import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


class ComputationalModels:
    def __init__(self, reward_means, reward_sd, model_type, condition="Gains"):
        """
        Initialize the DecayModel.

        Parameters:
        - num_options: Number of possible choices.
        """
        self.num_options = 4
        self.choices_count = np.zeros(self.num_options)
        self.condition = condition
        self.a = np.random.uniform(0, 1)  # Randomly set decay parameter between 0 and 1
        self.t = np.random.uniform(0, 5)
        self.b = np.random.uniform(-1, 1)
        self.iteration = 0

        if self.condition == "Gains":
            self.EVs = np.full(self.num_options, 0.5)
        elif self.condition == "Losses":
            self.EVs = np.full(self.num_options, -0.5)

        # Reward structure
        self.reward_means = reward_means
        self.reward_sd = reward_sd

        # Model type
        self.model_type = model_type

    def update(self, chosen, reward, trial):
        """
        Update EVs based on the choice, received reward, and trial number.

        Parameters:
        - chosen: Index of the chosen option.
        - reward: Reward received for the current trial.
        - trial: Current trial number.
        """
        if trial > 150:
            return self.EVs

        self.choices_count[chosen] += 1

        if self.model_type == 'decay':
            self.EVs = self.EVs * (1 - self.a)
            self.EVs[chosen] += reward

        elif self.model_type == 'decay_fre':
            self.EVs = self.EVs * (1 - self.a)
            multiplier = self.choices_count[chosen] ** self.b
            self.EVs[chosen] += reward * multiplier

        elif self.model_type == 'delta':
            prediction_error = reward - self.EVs[chosen]
            self.EVs[chosen] += self.a * prediction_error

        # print(f'C: {chosen}, R: {reward}, EV: {self.EVs}; it has been {self.choices_count[chosen]} times')

        return self.EVs

    def softmax(self, chosen, alt1):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen))
        denom = num + np.exp(min(700, c * alt1))
        return num / denom

    def simulate(self, num_trials=250, AB_freq=None, CD_freq=None, num_iterations=1,
                 beta_lower=-1, beta_upper=1):
        """
        Simulate the EV updates for a given number of trials and specified number of iterations.

        Parameters:
        - num_trials: Number of trials for the simulation.
        - AB_freq: Frequency of appearance for the AB pair in the first 150 trials.
        - CD_freq: Frequency of appearance for the CD pair in the first 150 trials.
        - num_iterations: Number of times to repeat the simulation.

        Returns:
        - A list of simulation results.
        """
        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.a = np.random.uniform()  # Randomly set decay parameter between 0 and 1
            self.t = np.random.uniform(0, 5)
            self.b = np.random.uniform(beta_lower, beta_upper)
            self.choices_count = np.zeros(self.num_options)

            EV_history = np.zeros((num_trials, self.num_options))
            trial_details = []
            trial_indices = []

            training_trials = [(0, 1), (2, 3)]
            training_trial_sequence = [training_trials[0]] * AB_freq + [training_trials[1]] * CD_freq
            np.random.shuffle(training_trial_sequence)

            # Distributing the next 100 trials equally among the four pairs (AC, AD, BC, BD)
            transfer_trials = [(2, 0), (1, 3), (0, 3), (2, 1)]
            transfer_trial_sequence = transfer_trials * 25
            np.random.shuffle(transfer_trial_sequence)

            for trial in range(num_trials):
                trial_indices.append(trial + 1)

                if trial < 150:
                    pair = training_trial_sequence[trial]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax(self.EVs[optimal], self.EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal
                else:
                    pair = transfer_trial_sequence[trial - 150]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax(self.EVs[optimal], self.EVs[suboptimal])
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                trial_details.append(
                    {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen)})
                reward = np.random.normal(self.reward_means[chosen], self.reward_sd[chosen])
                EV_history[trial] = self.update(chosen, reward, trial + 1)

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "a": self.a,
                "b": self.b,
                "t": self.t,
                "trial_details": trial_details,
                "EV_history": EV_history
            })

        return all_results

    def negative_log_likelihood(self, params, reward, choiceset, choice):
        """
        Compute the negative log likelihood for the given parameters and data.

        Parameters:
        - params: Parameters of the model.
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        - trial_numbers: List or array of trial numbers.
        """
        if self.model_type in ('decay', 'delta'):
            self.t = params[0]
            self.a = params[1]
        elif self.model_type == 'decay_fre':
            self.t = params[0]
            self.a = params[1]
            self.b = params[2]

        nll = 0
        self.choices_count = np.zeros(self.num_options)

        choiceset_mapping = {
            0: (0, 1),
            1: (2, 3),
            2: (2, 0),
            3: (2, 1),
            4: (0, 3),
            5: (1, 3)
        }

        trial = np.arange(1, 251)

        for r, cs, ch, trial in zip(reward, choiceset, choice, trial):
            cs_mapped = choiceset_mapping[cs]
            prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
            prob_choice_alt = self.softmax(self.EVs[cs_mapped[1]], self.EVs[cs_mapped[0]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)
            self.update(ch, r, trial)
        return nll

    def fit(self, data, num_iterations=1, beta_lower=-1, beta_upper=1):
        """
        Fit the model to the provided data.

        Parameters:
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        - trial_numbers: List or array of trial numbers.
        """

        all_results = []
        total_nll = 0  # Initialize the cumulative negative log likelihood
        total_n = len(data)  # Initialize the cumulative number of participants

        if self.model_type in ('decay', 'delta'):
            k = 2  # Initialize the cumulative number of parameters
        elif self.model_type == 'decay_fre':
            k = 3

        for participant_id, pdata in data.items():

            print(f"Fitting data for {participant_id}...")
            self.iteration = 0

            # Reset initial expected values for each participant
            if self.condition == "Gains":
                self.EVs = np.array([0.5, 0.5, 0.5, 0.5])
            elif self.condition == "Losses":
                self.EVs = np.array([-0.5, -0.5, -0.5, -0.5])

            best_nll = 100000  # Initialize best negative log likelihood to a large number
            best_initial_guess = None
            best_parameters = None

            for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

                self.iteration += 1
                print(f"\n=== Iteration {self.iteration} ===\n")

                if self.model_type in ('decay', 'delta'):
                    initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1)]
                    bounds = ((0, 5), (0, 1))
                elif self.model_type == 'decay_fre':
                    initial_guess = [np.random.uniform(0, 5), np.random.uniform(0, 1),
                                     np.random.uniform(beta_lower, beta_upper)]
                    bounds = ((0, 5), (0, 1), (beta_lower, beta_upper))

                result = minimize(self.negative_log_likelihood, initial_guess,
                                  args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_initial_guess = initial_guess
                    best_parameters = result.x

            total_nll += best_nll

            all_results.append({
                'participant_id': participant_id,
                'best_nll': best_nll,
                'best_initial_guess': best_initial_guess,
                'best_parameters': best_parameters,
                'total_nll': total_nll
            })

        # Compute the AIC and BIC
        aic = 2 * k + 2 * total_nll
        bic = k * np.log(total_n) + 2 * total_nll

        all_results.append({
            'AIC': aic,
            'BIC': bic
        })

        return all_results


def likelihood_ratio_test(null_results, alternative_results, df):
    """
    Perform a likelihood ratio test.

    Parameters:
    - null_nll: Negative log-likelihood of the simpler (null) model.
    - alternative_nll: Negative log-likelihood of the more complex (alternative) model.
    - df: Difference in the number of parameters between the two models.

    Returns:
    - p_value: p-value of the test.
    """
    # locate the nll values for the null and alternative models
    null_nll = null_results[0]['total_nll']
    alternative_nll = alternative_results[0]['total_nll']

    # Compute the likelihood ratio statistic
    lr_stat = 2 * (null_nll - alternative_nll)

    # Get the p-value
    p_value = chi2.sf(lr_stat, df)

    return p_value

