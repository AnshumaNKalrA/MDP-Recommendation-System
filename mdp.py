import operator
import pickle
import os
from tabulate import tabulate
import math # Ensure math is imported if your reward function uses it

from mdp_handler import MDPInitializer


class MDP:
    """
    Class to run the MDP.
    """

    def __init__(self, path='data', alpha=1, k=3, discount_factor=0.999, verbose=True, save_path="saved-models"):
        """
        The constructor for the MDP class.
        :param path: path to data
        :param alpha: the proportionality constant when considering transitions
        :param k: the number of items in each state
        :param discount_factor: the discount factor for the MDP
        :param verbose: flag to show steps
        :param save_path: the path to which models should be saved and loaded from
        """

        # Initialize the MDPInitializer
        self.mdp_i = MDPInitializer(path, k, alpha)
        self.df = discount_factor
        self.verbose = verbose
        self.save_path = save_path
        # The set of states
        self.S = {}
        # The set of state values
        self.V = {}
        # The set of actions
        self.A = []
        # The set of transitions
        self.T = {}
        # The policy of the MDP (best action for each state)
        self.policy = {}
        # A policy list (ranked actions for each state)
        self.policy_list = {}
        self.k = k # Store k value

    def print_progress(self, message):
        if self.verbose:
            print(message)

    def initialise_mdp(self):
        """
        The method to initialise the MDP.
        :return: None
        """

        # Initialising the actions
        self.print_progress("Getting set of actions.")
        self.A = self.mdp_i.actions
        self.print_progress("Set of actions obtained.")

        # Initialising the states, state values, policy
        self.print_progress("Getting states, state-values, policy.")
        self.S, self.V, self.policy, self.policy_list = self.mdp_i.generate_initial_states()
        self.print_progress("States, state-values, policy obtained.")

        # Initialise the transition table
        self.print_progress("Getting transition table.")
        self.T = self.mdp_i.generate_transitions(self.S, self.A)
        self.print_progress("Transition table obtained.")

    def one_step_lookahead(self, state):
        """
        Helper function to calculate state-value function.
        :param state: state to consider
        :return: action values for that state
        """

        # Initialise the action values and set to 0
        action_values = {}
        for action in self.A:
            action_values[action] = 0

        # Calculate the action values for each action
        if state not in self.T: # Add a check here
            # print(f"Warning: State {state} not found in transition table.")
            return {action: 0 for action in self.A} # Return 0 values or handle differently

        for action in self.A:
            if action not in self.T[state]: # Add a check here
                 continue # No transitions for this action from this state

            for next_state, P_and_R in self.T[state][action].items():
                if next_state not in self.V:
                    # print(f"Warning: Next state {next_state} not found in V. Initializing to 0.")
                    self.V[next_state] = 0 # Initialize value if next_state is new
                # action_value +=  probability * (reward + (discount * next_state_value))
                action_values[action] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[next_state]))

        return action_values

    def update_policy(self):
        """
        Helper function to update the policy based on the value function.
        :return: None
        """

        for state in self.S: # Only update policy for known states
            action_values = self.one_step_lookahead(state)

            # Ensure action_values is not empty before calling max
            if not action_values:
                 # Handle case where no action values could be computed for the state
                 # print(f"Warning: No action values computed for state {state}. Skipping policy update for this state.")
                 continue

            # The action with the highest action value is chosen
            self.policy[state] = max(action_values.items(), key=operator.itemgetter(1))[0]
            self.policy_list[state] = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)

    def policy_eval(self):
        """
        Helper function to evaluate a policy
        :return: estimated value of each state following the policy and state-value
        """

        # Initialise the policy values
        policy_value = {}
        # Only evaluate for states where a policy is defined
        for state in self.policy:
             policy_value[state] = 0

        # Find the policy value for each state and its respective action dictated by the policy
        for state, action in self.policy.items():
            if state not in self.T or action not in self.T[state]:
                 # print(f"Warning: State {state} or action {action} not in transitions T during policy evaluation.")
                 continue # Skip if transition data is missing for this state-action pair

            for next_state, P_and_R in self.T[state][action].items():
                if next_state not in self.V:
                    # print(f"Warning: Next state {next_state} not found in V during policy evaluation. Initializing to 0.")
                    self.V[next_state] = 0
                # policy_value +=  probability * (reward + (discount * next_state_value))
                policy_value[state] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[next_state]))

        return policy_value

    def compare_policy(self, policy_prev):
        """
        Helper function to compare the given policy with the current policy
        :param policy_prev: the policy to compare with
        :return: a boolean indicating if the policies are different or not
        """

        for state in policy_prev:
            # If the policy does not match even once then return False
            # Also check if the state exists in the current policy
            if state not in self.policy or policy_prev[state] != self.policy[state]:
                return False
        return True

    def policy_iteration(self, max_iteration=1000, start_where_left_off=False, to_save=True):
        """
        Algorithm to solve the MDP
        :param max_iteration: maximum number of iterations to run.
        :param start_where_left_off: flag to load a previous model(set False if not and filename otherwise)
        :param to_save: flag to save the current model
        :return: None
        """

        # Load a previous model
        if start_where_left_off:
            self.load(start_where_left_off)

        # Start the policy iteration
        policy_prev = self.policy.copy()
        for i in range(max_iteration):
            self.print_progress(f"Iteration {i+1}:")

            # Evaluate given policy
            self.V = self.policy_eval()

            # Improve policy
            self.update_policy()

            # If the policy not changed over 10 iterations it converged
            if i % 10 == 0: # Check every 10 iterations
                if self.compare_policy(policy_prev):
                    self.print_progress(f"Policy converged at iteration {i+1}")
                    break
                policy_prev = self.policy.copy()
        else: # This else block runs if the loop finishes without breaking
             self.print_progress(f"Policy iteration finished after {max_iteration} iterations without convergence.")


        # Save the model
        if to_save:
            self.save(f"mdp-model_k={self.mdp_i.k}.pkl") # Use self.mdp_i.k


    def save(self, filename):
        """
        Method to save the trained model
        :param filename: the filename it should be saved as
        :return: None
        """

        full_path = os.path.join(self.save_path, filename)
        self.print_progress(f"Saving model to {full_path}")
        os.makedirs(self.save_path, exist_ok=True)
        try:
            with open(full_path, 'wb') as f:
                # Only pickle the necessary attributes for the trained model
                # Include k, policy_list, and mdp_i (which contains games, actions etc.)
                pickle.dump({
                    'k': self.k, # Save k
                    'policy_list': self.policy_list,
                    # You might need other attributes from mdp_i, depending on what recommend needs
                    'mdp_i_games': self.mdp_i.games, # Needed for title lookup
                    'mdp_i_actions': self.mdp_i.actions # Might be needed
                    # Add other mdp_i attributes if needed by recommend or other methods after loading
                    # e.g., 'mdp_i_transactions': self.mdp_i.transactions # If recommend uses this
                }, f, pickle.HIGHEST_PROTOCOL)
            self.print_progress("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model to {full_path}: {e}")


    def load(self, filename):
        """
        Method to load a previous trained model
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        full_path = os.path.join(self.save_path, filename)
        self.print_progress(f"Loading model from {full_path}")
        try:
            with open(full_path, 'rb') as f:
                # Load the saved attributes
                loaded_data = pickle.load(f)

            # Assign loaded attributes back to the instance
            self.k = loaded_data.get('k', self.k) # Load k, use default if not found
            self.policy_list = loaded_data.get('policy_list', {}) # Load policy_list
            # Re-initialize mdp_i or assign its components if saved separately
            # Assuming mdp_i is initialized with path and k in __init__
            # We might need to update its internal games/actions if they were saved separately
            if 'mdp_i_games' in loaded_data:
                 # If mdp_i exists, update its games attribute
                 if hasattr(self, 'mdp_i'):
                      self.mdp_i.games = loaded_data['mdp_i_games']
                 else:
                      # If mdp_i wasn't created (e.g., loading a model without initialising first),
                      # this is a potential issue. Ensure MDP is instantiated correctly.
                      print("Warning: mdp_i not available during model loading. Cannot update games.")

            if 'mdp_i_actions' in loaded_data:
                 if hasattr(self, 'mdp_i'):
                      self.mdp_i.actions = loaded_data['mdp_i_actions']
                 else:
                      print("Warning: mdp_i not available during model loading. Cannot update actions.")

            # Rebuild the simpler policy dictionary from policy_list if needed by other methods
            # This might not be strictly necessary if policy_list is the primary source of policy
            # self.policy = {state: ranked_actions[0][0] for state, ranked_actions in self.policy_list.items() if ranked_actions}

            self.print_progress("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error loading model: File not found at {full_path}")
            raise # Re-raise the exception if file not found
        except Exception as e:
            print(f"Error loading model from {full_path}: {e}")
            raise # Re-raise the exception after printing

    def save_policy(self, filename):
        """
        Method to save the policy_list
        :param filename: the filename it should be saved as
        :return: None
        """

        full_path = os.path.join(self.save_path, filename)
        self.print_progress(f"Saving policy list to {full_path}")
        os.makedirs(self.save_path, exist_ok=True)
        try:
            with open(full_path, 'wb') as f:
                pickle.dump(self.policy_list, f, pickle.HIGHEST_PROTOCOL)
            self.print_progress("Policy list saved successfully.")
        except Exception as e:
            print(f"Error saving policy list to {full_path}: {e}")


    def load_policy(self, filename):
        """
        Method to load a previous policy_list
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        full_path = os.path.join(self.save_path, filename)
        self.print_progress(f"Loading policy list from {full_path}")
        try:
            with open(full_path, 'rb') as f:
                self.policy_list = pickle.load(f)
            self.print_progress("Policy list loaded successfully.")
        except FileNotFoundError:
            print(f"Error loading policy list: File not found at {full_path}")
            raise
        except Exception as e:
            print(f"Error loading policy list from {full_path}: {e}")
            raise

    # Existing recommend method based on user_id
    def recommend(self, user_id):
        """
        Method to provide recommendation to the user based on their loaded history.
        :param user_id: the user_id of a given user
        :return: a list of (game_title, score) tuples
        """
        # self.print_progress("Recommending for " + str(user_id))

        # Ensure mdp_i and transactions are loaded
        if not hasattr(self.mdp_i, 'transactions') or user_id not in self.mdp_i.transactions:
             raise KeyError(f"User ID {user_id} not found in loaded transactions.")

        # Get the user's transaction history (which contains IDs based on MDPInitializer)
        user_transactions_ids = self.mdp_i.transactions[user_id]

        # Pad with None for history shorter than k
        padded_transactions = [None] * (self.k - 1) + user_transactions_ids

        # Form the state tuple from the last k items (IDs or None)
        # The state is a tuple of the last k game IDs (or None)
        user_state_tuple = tuple(padded_transactions[-self.k:])

        # Now use the new method to get recommendations based on this state tuple
        # Pass the ID-to-Title map so the new method can return titles
        return self.recommend_by_state(user_state_tuple, game_id_to_title_map=self.mdp_i.games)


    # New method to recommend based on a given state tuple
    def recommend_by_state(self, state_tuple, game_id_to_title_map):
        """
        Method to provide recommendation based on a specific state tuple.
        :param state_tuple: A tuple representing the last k game IDs (or None).
        :param game_id_to_title_map: A dictionary mapping game IDs to game titles.
        :return: a list of (game_title, score) tuples.
                 Returns an empty list if the state is not in the policy_list.
        """
        # Ensure policy_list is loaded
        if not self.policy_list:
             print("Warning: policy_list is empty. Cannot recommend by state.")
             return []

        # Look up the state tuple in the trained policy list
        if state_tuple not in self.policy_list:
            # print(f"Warning: State tuple {state_tuple} not found in policy_list.")
            return [] # Return empty list if the state was not encountered during training

        # Get the ranked list of recommended actions (game ID, score) for this state
        ranked_recommendations_ids_scores = self.policy_list[state_tuple]

        rec_list_titles_scores = []
        # Convert game IDs to titles using the provided map
        for game_id, score in ranked_recommendations_ids_scores:
            # Look up the game title using the ID
            game_title = game_id_to_title_map.get(game_id, f"Unknown Game ID: {game_id}") # Use get with default

            rec_list_titles_scores.append((game_title, score))

        return rec_list_titles_scores


    # Existing evaluation methods (kept for completeness, not directly used by frontend)
    def evaluate_decay_score(self, alpha=10):
        """
        Method to evaluate the given MDP using exponential decay score
        :param alpha: a parameter in exponential decay score
        :return: the average score
        """
        # This method needs access to original transactions from mdp_i
        if not hasattr(self.mdp_i, 'transactions'):
             print("Error: Transactions data not loaded in mdp_i. Cannot evaluate.")
             return 0.0

        transactions = self.mdp_i.transactions.copy()

        user_count = 0
        total_score = 0
        # Generating a testing for each test case
        for user in transactions:
            total_list = len(transactions[user])
            if total_list <= self.k: # Need at least k+1 transactions to form a state and a next item
                continue

            score = 0
            # Iterate through transactions to create states and predict the next one
            for i in range(self.k -1, total_list - 1): # Start index to ensure at least k items before the next
                 # Form the state tuple from the last k items (IDs)
                 user_state = tuple(transactions[user][i - (self.k - 1) : i + 1])

                 # Get recommendations for this state using the new method
                 # Need game_id_to_title_map, but recommend_by_state needs IDs in state_tuple input
                 # We'll get the recommendations in terms of Titles and then find the rank of the actual next ID's title
                 # This requires the game_id to title map here as well, or modify recommend_by_state to return IDs
                 # Let's assume we get titles back and convert the actual next ID to title to find its rank
                 actual_next_game_id = transactions[user][i + 1]
                 actual_next_game_title = self.mdp_i.games.get(actual_next_game_id, f"Unknown_{actual_next_game_id}") # Lookup title


                 rec_list = self.recommend_by_state(user_state, game_id_to_title_map=self.mdp_i.games) # Get recommendations as (title, score)
                 rec_titles = [rec[0] for rec in rec_list] # Extract just the titles

                 try:
                     # Find the rank (position) of the actual next game's title in the recommended list
                     m = rec_titles.index(actual_next_game_title) + 1
                     score += 2 ** ((1 - m) / (alpha - 1))
                 except ValueError:
                      # If the actual next game was not in the recommended list
                      pass # Score remains unchanged for this step

            # Check if any steps were scored before dividing
            num_scored_steps = total_list - 1 - (self.k - 1)
            if num_scored_steps > 0:
                score /= num_scored_steps
                total_score += 100 * score
                user_count += 1
            # else: # User had too few transactions to create a state and next item
                 # print(f"Skipping evaluation for user {user} due to insufficient transactions ({total_list} < {self.k + 1})")


        return total_score / user_count if user_count > 0 else 0.0


    def evaluate_recommendation_score(self, m=10):
        """
        Function to evaluate the given MDP using recommendation score (Hit Rate @ m).
        :param m: The rank cutoff.
        :return: the average score (percentage of hits within top m).
        """
         # This method needs access to original transactions from mdp_i
        if not hasattr(self.mdp_i, 'transactions'):
             print("Error: Transactions data not loaded in mdp_i. Cannot evaluate.")
             return 0.0

        transactions = self.mdp_i.transactions.copy()


        user_count = 0
        total_score = 0
        # Generating a testing for each test case
        for user in transactions:
            total_list = len(transactions[user])
            if total_list <= self.k: # Need at least k+1 transactions to form a state and a next item
                continue

            item_count = 0
            # Iterate through transactions to create states and predict the next one
            for i in range(self.k - 1, total_list - 1): # Start index to ensure at least k items before the next
                 # Form the state tuple from the last k items (IDs)
                 user_state = tuple(transactions[user][i - (self.k - 1) : i + 1])

                 # Get recommendations for this state using the new method
                 actual_next_game_id = transactions[user][i + 1]
                 actual_next_game_title = self.mdp_i.games.get(actual_next_game_id, f"Unknown_{actual_next_game_id}") # Lookup title

                 rec_list = self.recommend_by_state(user_state, game_id_to_title_map=self.mdp_i.games) # Get recommendations as (title, score)
                 rec_titles = [rec[0] for rec in rec_list] # Extract just the titles


                 try:
                     # Find the rank (position) of the actual next game's title in the recommended list
                     rank = rec_titles.index(actual_next_game_title) + 1
                     if rank <= m:
                         item_count += 1
                 except ValueError:
                      # If the actual next game was not in the recommended list
                      pass # Not a hit

            # Check if any steps were evaluated before dividing
            num_evaluated_steps = total_list - 1 - (self.k - 1)
            if num_evaluated_steps > 0:
                score = item_count / num_evaluated_steps
                total_score += 100 * score
                user_count += 1
            # else: # User had too few transactions to create a state and next item
                 # print(f"Skipping evaluation for user {user} due to insufficient transactions ({total_list} < {self.k + 1})")


        return total_score / user_count if user_count > 0 else 0.0


if __name__ == '__main__':
    # Example usage (CLI)
    rs = MDP(path='data-mini', k=2) # Specify k=2 or k=3
    # Make sure you have run policy_iteration and saved the model before running this
    # rs.initialise_mdp() # Uncomment and run this and policy_iteration if you need to train
    # rs.policy_iteration() # Uncomment to train

    try:
        rs.load(f'mdp-model_k={rs.k}.pkl')
        print(f"Loaded model for k={rs.k}")
        headers = ['Rank', 'Game', 'Score']
        while True:
            u = input("Enter a user ID (or 'custom' for history input, 'quit' to exit): ")
            if u.lower() == 'quit':
                 break
            elif u.lower() == 'custom':
                 history_input = input(f"Enter the last {rs.k} game IDs or titles, comma-separated (most recent last): ")
                 history_list = [item.strip() for item in history_input.split(',') if item.strip()]

                 # Need to convert input titles/IDs to IDs and form state tuple
                 # This requires a map from title to ID or assuming input is IDs
                 # Let's assume for this CLI example, the user inputs IDs
                 # For frontend, we handle title-to-ID conversion in app.py

                 if len(history_list) > rs.k:
                      print(f"Warning: Input history longer than k={rs.k}. Using the last {rs.k} items.")
                      state_ids = history_list[-rs.k:]
                 elif len(history_list) < rs.k:
                      # Pad with None if history is shorter than k
                      padding = [None] * (rs.k - len(history_list))
                      state_ids = padding + history_list
                 else:
                      state_ids = history_list

                 state_tuple = tuple(state_ids)

                 print(f"Using state tuple (IDs): {state_tuple}")

                 # Need the ID to Title map to display results
                 if not hasattr(rs.mdp_i, 'games'):
                      print("Error: Game ID to Title map not loaded. Cannot display recommendations.")
                      continue

                 # Get recommendations using the new method
                 custom_r_list = rs.recommend_by_state(state_tuple, game_id_to_title_map=rs.mdp_i.games)

                 if custom_r_list:
                     print("\nRecommendations:")
                     print(tabulate([[ind+1, r[0], r[1]] for ind, r in enumerate(custom_r_list)], headers, "psql"))
                 else:
                     print(f"No recommendations found for state {state_tuple}. This state may not have been in the training data.")

            else: # Handle user_id input
                 try:
                     r_list = rs.recommend(u)
                     if r_list:
                        print(f"\nRecommendations for User {u}:")
                        print(tabulate([[ind+1, r[0], r[1]] for ind, r in enumerate(r_list)], headers, "psql"))
                     else:
                         print(f"No recommendations found for user {u}. This user might not have transactions or their state is not in the policy.")
                 except KeyError as e:
                     print(e) # Print the specific error message
                 except Exception as e:
                     print(f"An error occurred: {e}")


    except FileNotFoundError:
        print(f"Model file not found for k={rs.k}. Please train the model first by uncommenting the lines above.")
    except Exception as e:
         print(f"An error occurred during loading or execution: {e}")