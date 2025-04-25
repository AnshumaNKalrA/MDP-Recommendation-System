from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import csv
import json
from mdp import MDP
import random
import os
import math 

# Configure Flask to serve static files from the 'static' folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app) # Enable CORS for all routes

# Define the path to your data directory
DATA_PATH = './data-mini'
# Define the expected path for your trained MDP model file
MODEL_SAVE_DIR = 'saved-models' # Directory where models are saved
MODEL_FILENAME = 'mdp-model_k=2.pkl' # Assuming k=2 for the loaded model
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)


# --- Start: Load or Train MDP Model ---
# Initialize rs to None. We will attempt to load or train the model next.
rs = None
MDP_K_VALUE = 2 # Define the k value your model expects (based on filename)

try:
    # Create an MDP instance. Pass the expected k value.
    # This also initializes the MDPInitializer which reads data like games.csv and transactions.csv internally.
    rs = MDP(path=DATA_PATH, k=MDP_K_VALUE, save_path=MODEL_SAVE_DIR)

    # Ensure the directory for saving/loading the model exists
    os.makedirs(rs.save_path, exist_ok=True)

    # Check if the trained model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}.")
        print("Attempting to initialize and train the MDP model...")
        try:
            # Initialize the MDP (generate states, transitions, etc.)
            rs.initialise_mdp()

            # Check if states were successfully generated
            if not rs.S:
                 raise RuntimeError("MDP states were not generated during initialisation. Check data files and MDPInitializer.")

            # Train the policy using policy iteration. This should save the model.
            rs.policy_iteration(to_save=True) # Ensure saving is enabled
            print("MDP training complete. Model saved.")

        except Exception as e:
            print(f"Error during initial MDP training: {e}")
            print("Please ensure your MDP training setup in mdp.py is correct and can generate and save the model.")
            print(f"Attempted to save model to: {MODEL_PATH}")
            # If training fails, rs might be partially initialized, but the model file won't exist.
            # The subsequent load attempt will likely fail or rs might not be fully functional.


    # Attempt to load the model regardless of whether training just happened or it already existed.
    if os.path.exists(MODEL_PATH):
        rs.load(MODEL_FILENAME) # Load the trained policy and other MDP data
        print(f"Successfully loaded MDP model from {MODEL_PATH}")
    else:
        # If the model still doesn't exist after attempting training, something went wrong.
        print(f"Error: Model file {MODEL_PATH} not found after attempting training. Cannot proceed without model.")
        rs = None # Ensure rs is None if loading is not possible.


except FileNotFoundError:
     print(f"Error: Data files (e.g., games.csv, users.csv, transactions.csv) not found in {DATA_PATH}. Please check DATA_PATH.")
     rs = None # Ensure rs is None if data files are missing.
except Exception as e:
    print(f"An unexpected error occurred during MDP initialization or loading: {e}")
    print("Please check your MDP code (mdp.py, mdp_handler.py) and data files.")
    rs = None # Ensure rs is None on other errors.

# --- End: Load or Train MDP Model ---


# --- Start: Load Game Mappings (ID<->Title) ---
# These mappings are needed for converting between frontend titles and backend IDs.
GAME_ID_TO_TITLE = {}
TITLE_TO_GAME_ID = {}
# Access the mapping from the MDPInitializer instance, which is created when MDP is instantiated.
# This is only possible if the 'rs' object was successfully created and initialized.
if rs is not None and hasattr(rs, 'mdp_i') and hasattr(rs.mdp_i, 'games'):
    GAME_ID_TO_TITLE = rs.mdp_i.games
    # Create the reverse mapping (Title to ID)
    TITLE_TO_GAME_ID = {title: game_id for game_id, title in GAME_ID_TO_TITLE.items()}
    print("Successfully loaded Game ID and Title mappings from MDPInitializer.")
elif rs is None:
     print("Warning: MDP (rs) could not be initialized, cannot load Game ID/Title mappings.")
else:
     # This case might occur if MDP was initialized but failed before mdp_i or games was populated.
     print("Warning: rs.mdp_i or rs.mdp_i.games not available after MDP initialization. Cannot load Game ID/Title mappings.")
# --- End: Load Game Mappings ---


# --- Start: Load Game Thumbnail Mapping ---
def load_game_thumbnails(games_csv_path):
    """
    Loads a mapping from game-title to the URL path of its thumbnail image (based on game-id).
    Assumes images are in ./static/thumbnails/ and named [game-id].jpg.
    Correctly constructs the URL path for the frontend using game-id.
    """
    game_thumbnails = {}
    try:
        with open(games_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Assuming games.csv has 'game-id' and 'game-title' columns
            for row in reader:
                game_id = row.get('game-id')
                game_title = row.get('game-title') # Get the actual game title

                if game_id and game_title:
                    # Construct the expected URL path for the frontend based on game-id
                    image_filename = f'{game_id}.jpg'
                    # This is the URL path that the frontend will request images from
                    image_url_path = f'/static/thumbnails/{image_filename}'

                    # Map the game title (used by the frontend) to the image URL path
                    game_thumbnails[game_title] = image_url_path
                # Handle rows with missing 'game-id' or 'game-title' if necessary
                # print(f"Warning: Skipping row due to missing data: {row}")

        # Optional: Add a placeholder path if you have a placeholder image in static/
        # For example, if you put placeholder.jpg in ./static/
        # If a game title isn't found in this map in the frontend, it can fall back to a default URL like /static/placeholder.jpg
        # This map provides specific image paths where available.


    except FileNotFoundError:
        print(f"Error: games.csv not found at {games_csv_path}")
    except Exception as e:
        print(f"Error loading game thumbnails from {games_csv_path}: {e}")
    return game_thumbnails

# Construct the full path to games.csv using DATA_PATH
games_csv_full_path = os.path.join(DATA_PATH, 'games.csv')

# Load thumbnails *after* defining the function and getting the full path
GAME_THUMBNAILS = load_game_thumbnails(games_csv_full_path)
# --- End: Load Game Thumbnail Mapping ---


# --- Start: Load Initial User Data (Transactions & Recommendations for sample users) ---
# Pass the game_id_to_title_map to this function
def get_user_data(users_csv_path, transactions_csv_path, num_users=25, game_id_to_title_map=None):
    """
    Fetches user data (transactions and recommendations) for a limited number of users.
    Converts transaction game IDs (from transactions.csv 'game-title' column) to titles
    using the provided game_id_to_title_map.
    """
    users_data = {} # Use a dictionary keyed by user_id initially for easy access
    all_users = [] # To hold all users from users.csv

    try:
        # Read users from users.csv
        with open(users_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_users = list(reader) # Read all user rows into a list

        # Select a limited number of random users from the list
        if not all_users:
             print("Warning: No users found in users.csv. Cannot load user data.")
             return [] # Return empty list if no users are found

        # Select random users, ensuring we don't ask for more than exist
        selected_users = random.sample(all_users, min(num_users, len(all_users)))

        # Initialize user data structures for selected users
        for row in selected_users:
            user_id = row.get('user-id')
            if user_id: # Ensure user_id is not None or empty
                 users_data[user_id] = {'userId': user_id, 'transactions': [], 'recommendations': []}

        # If no users were selected (e.g., input num_users was 0 or users.csv was empty), return early
        if not users_data:
             print("No users selected for initial data load.")
             return []

        # Read transactions from transactions.csv
        with open(transactions_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get('user-id')
                # Read the value from the 'game-title' column, which we know is the game ID
                game_id_from_csv = row.get('game-title')
                # Optional: Read other columns if needed later
                # behaviour = row.get('behaviour-name')
                # value = row.get('values')

                # Process transaction only for selected users and if data is valid
                if user_id and game_id_from_csv and user_id in users_data:
                    # --- Convert Game ID (from CSV) to Game Title ---
                    # Use the provided map to look up the actual game title
                    # Provide a default value (like None) if the key isn't found in the map
                    game_title = game_id_to_title_map.get(game_id_from_csv) if game_id_to_title_map else None

                    if game_title: # Check if the lookup was successful (ID found in games.csv mapping)
                         # Add the game title to the user's transactions list if not already present
                         if game_title not in users_data[user_id]['transactions']:
                             users_data[user_id]['transactions'].append(game_title)
                    else:
                        # Handle cases where a game ID in transactions.csv is not found in the games.csv mapping
                        # print(f"Warning: Game ID '{game_id_from_csv}' for user {user_id} in transactions.csv not found in games.csv mapping. Skipping this transaction.")
                        pass # Silently skip transactions with unknown game IDs


        # Get recommendations for the selected users
        # This part assumes rs.recommend takes a user_id and internally handles
        # looking up their history (presumably using the original ID-based history loaded into rs.mdp_i.transactions)
        # and returns a list of (game_title, score) tuples.
        if rs is not None and hasattr(rs, 'recommend'):
            for user_id in users_data:
                 # Note: The rs.recommend method now uses recommend_by_state which returns titles
                try:
                    r_list = rs.recommend(user_id) # Call the recommend method on the loaded MDP instance
                    # The output r_list is expected to be a list of (title, score) tuples
                    # Convert this list of tuples into the desired list of dictionaries for the frontend
                    users_data[user_id]['recommendations'] = [{"title": rec[0], "score": rec[1]} for rec in r_list]
                except KeyError as e:
                     # This might happen if the user_id exists but their specific transaction history state
                     # is not present in the trained MDP policy (based on ID history).
                     print(f"Warning: {e}") # Print the specific KeyError message from recommend
                     users_data[user_id]['recommendations'] = [] # Assign empty list
                except Exception as e:
                     print(f"Error getting recommendations for user {user_id}: {e}")
                     users_data[user_id]['recommendations'] = [] # Assign empty list


        else:
            print("Warning: MDP recommender (rs) not successfully initialized or loaded. Cannot generate recommendations for selected users.")


    except FileNotFoundError as e:
        print(f"Error loading user or transaction data: {e}")
        # Return an empty list if the necessary data files are not found
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing user or transaction data: {e}")
        # Return an empty list on other processing errors
        return []

    # Return the list of user data dictionaries (converted from the users_data dictionary's values)
    # This ensures the API returns a JSON array.
    return list(users_data.values())
# --- End: Load Initial User Data ---


# Construct full paths for user and transaction CSVs using DATA_PATH
users_csv_full_path = os.path.join(DATA_PATH, 'users.csv')
transactions_csv_full_path = os.path.join(DATA_PATH, 'transactions.csv')

# Load initial user data on app startup
# This must happen *after* the MDP model (rs) and the Game ID to Title mapping are loaded.
USER_DATA = [] # Initialize USER_DATA as an empty list
# Only attempt to load USER_DATA if both the MDP model (rs) and the GAME_ID_TO_TITLE map are available.
if rs is not None and GAME_ID_TO_TITLE:
     # Call get_user_data, passing the full paths and the game_id_to_title_map
     USER_DATA = get_user_data(users_csv_full_path, transactions_csv_full_path, num_users=25, game_id_to_title_map=GAME_ID_TO_TITLE)
     # print(f"Initial USER_DATA loaded: {USER_DATA}") # Optional debug print
elif rs is None:
     print("Skipping initial USER_DATA load because MDP model (rs) failed to load.")
elif not GAME_ID_TO_TITLE:
     print("Skipping initial USER_DATA load because Game ID to Title map could not be loaded.")


# --- Define API Endpoints ---

@app.route('/api/users')
def get_users():
    """
    API endpoint to return data for the initially selected users.
    Returns a JSON array of user objects.
    """
    # USER_DATA is already loaded on app startup and should be a list of dictionaries
    return jsonify(USER_DATA)


@app.route('/api/recommendations/<user_id>')
def get_recommendations(user_id):
    """
    API endpoint to return recommendations for a specific user ID.
    This endpoint can be used to fetch recommendations dynamically,
    though the initial data load already includes them for the selected users.
    """
    # Ensure the recommender system (rs) is available
    if rs is None or not hasattr(rs, 'recommend'):
         return jsonify({"error": "Recommender system not initialized"}), 500

    # This endpoint uses the existing rs.recommend(user_id) method.
    # That method internally looks up the user's original history (ID-based)
    # and uses the new recommend_by_state method to get title-based recommendations.

    try:
        # Call the recommend method with the user_id.
        # This should return a list of (title, score) tuples.
        r_list = rs.recommend(user_id)
        # Convert the list of tuples into a list of dictionaries for the JSON response
        recommendations = [{"title": rec[0], "score": rec[1]} for rec in r_list]
        return jsonify(recommendations)

    except KeyError as e:
        # rs.recommend might raise KeyError if the user_id is not found in the internal data,
        # or if their specific history sequence isn't a state in the trained policy.
        print(f"Error: {e} during /api/recommendations call.") # Print the specific error message from recommend
        return jsonify({"error": f"Recommendations not available for user {user_id}"}), 404
    except Exception as e:
        # Catch any other potential errors during recommendation generation
        print(f"Error fetching recommendations for user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/game_thumbnails')
def get_game_thumbnails():
    """
    API endpoint to return the mapping of game titles to their thumbnail image URL paths.
    Returns a JSON object.
    """
    # GAME_THUMBNAILS is loaded on app startup and should be a dictionary
    return jsonify(GAME_THUMBNAILS)


# --- New API Endpoint for Custom History Recommendations ---
@app.route('/api/recommendations/custom', methods=['POST'])
def get_custom_recommendations():
    """
    API endpoint to return recommendations based on a custom list of game titles (history).
    Expects a POST request with a JSON body like:
    { "history": ["Game Title 1", "Game Title 2", ...] }
    Uses the last k titles from the history to form the state tuple and get recommendations.
    """
    # Ensure the recommender system (rs) and necessary mappings are available
    if rs is None or not hasattr(rs, 'recommend_by_state') or not TITLE_TO_GAME_ID or not GAME_ID_TO_TITLE:
         return jsonify({"error": "Recommender system or mappings not initialized"}), 500

    # Get the JSON data from the request body
    request_data = request.get_json()

    # Check if the 'history' key exists and is a list
    if not request_data or 'history' not in request_data or not isinstance(request_data['history'], list):
        return jsonify({"error": "Invalid request body. Expected JSON with a 'history' key containing a list of game titles."}), 400

    # Get the custom history list of titles
    custom_history_titles = request_data['history']
    print(f"Received custom history (titles): {custom_history_titles}")

    # Convert titles to game IDs
    custom_history_ids = []
    unknown_titles = []
    for title in custom_history_titles:
        game_id = TITLE_TO_GAME_ID.get(title)
        if game_id:
            custom_history_ids.append(game_id)
        else:
            unknown_titles.append(title)
            print(f"Warning: Unknown game title in custom history: {title}")

    if unknown_titles:
         # Optional: Return an error or a warning if unknown titles were included
         # return jsonify({"error": f"One or more game titles in history were not recognized: {', '.join(unknown_titles)}"}), 400
         pass # For now, we'll just skip unknown titles

    # Get the required history length (k) from the MDP instance
    mdp_k = rs.k # Use the k value from the loaded MDP

    # Take the last k game IDs from the converted history
    # Pad with None if the history is shorter than k
    if len(custom_history_ids) < mdp_k:
        padding = [None] * (mdp_k - len(custom_history_ids))
        state_ids_for_tuple = padding + custom_history_ids
    else:
        state_ids_for_tuple = custom_history_ids[-mdp_k:]

    # Form the state tuple (must be a tuple)
    state_tuple = tuple(state_ids_for_tuple)
    print(f"Generated state tuple (IDs): {state_tuple}")

    try:
        # Get recommendations using the recommend_by_state method
        # Pass the ID-to-Title map as it's needed by that method
        recommendations_list = rs.recommend_by_state(state_tuple, game_id_to_title_map=GAME_ID_TO_TITLE)

        if recommendations_list:
             # Format the recommendations for the frontend
             recommendations_formatted = [{"title": rec[0], "score": rec[1]} for rec in recommendations_list]
             print(f"Recommendations found: {len(recommendations_formatted)}")
             return jsonify(recommendations_formatted)
        else:
             print(f"No recommendations found for state {state_tuple}.")
             # Return an empty list if the state is not in the policy
             return jsonify([]), 200 # Return 200 OK with an empty list


    except Exception as e:
        # Catch any other potential errors during recommendation generation
        print(f"Error getting custom recommendations for state {state_tuple}: {e}")
        return jsonify({"error": "Internal server error generating recommendations"}), 500


# @app.route('/api/game_thumbnails')
# def get_game_thumbnails():
#     """
#     API endpoint to return the mapping of game titles to their thumbnail image URL paths.
#     Returns a JSON object.
#     """
#     # GAME_THUMBNAILS is loaded on app startup and should be a dictionary
#     return jsonify(GAME_THUMBNAILS)


# Define the route to serve the main index.html file
# This makes the frontend accessible when you visit the root URL (/)
@app.route('/')
def serve_index():
    """
    Serves the index.html file from the current directory.
    """
    # Assuming index.html is located in the same directory as app.py
    # send_from_directory is used to safely serve files from a specified directory
    return send_from_directory('.', 'index.html')

# Flask automatically handles serving static files from the 'static' folder
# based on the 'static_folder' and 'static_url_path' parameters set when
# the Flask app instance was created. So, requests to /static/... will be
# served from the ./static/ directory.


# --- Run the Flask App ---
# This block executes only when the script is run directly (e.g., python app.py)
if __name__ == '__main__':
    # Run the Flask development server.
    # debug=True enables helpful debugging features, including automatic reloading.
    # It should NOT be used in production.
    # Set threaded=False or processes > 1 if needed for concurrency,
    # but development server is typically single-threaded.
    app.run(debug=True)