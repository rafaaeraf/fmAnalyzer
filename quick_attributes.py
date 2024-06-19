from run import *

INPUT_FILE = "C:\\Users\\rafaa\\Documents\\Sports Interactive\\Football Manager 2020\\\Indefinido.html"

# Get current directory and read attributes weights CSVs
curr_dir = os.path.dirname(os.path.abspath(__file__))
attributes = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "attributes.csv"))
positions = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "positions.csv"))
weights = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "weights.csv"),
                        index_col="short_name")


# Read html
players = pd.read_html(INPUT_FILE, header=0, encoding="utf-8", na_values=["-", "- -"])
players = players[0]

# Get df_type
players["df_type"] = get_df_type(players)

# Parse players positions - Go from "MD, M/MO (DC), PL (C)" to MD,M(D),M(C),MO(D),MO(C),PL(C)
players["edited_positions"] = players["Posição"].apply(parse_position)

# Classify player/coach on attribute availability - complete, complete but range,
# incomplete, all incomplete
players["data_completion"] = players.apply(lambda x: check_data_completion(x, attributes), axis=1)

# Handle missing data for all players that are "complete but range" or "incomplete"
players = players.apply(lambda x: handle_missing_attributes(x, attributes, None), axis=1)

# Convert attributes to numeric
players = convert_df_columns_to_numeric(
    players, attributes.loc[attributes["player_coach"] == "player", "short_name"])

# Add general roles to positions and weights df
positions, weights = create_general_roles(players, positions, weights)

# Replace the attribute colors with weight values
player_roles_full_name = positions.loc[positions["positions"].notna(), "full_name"].unique()
weights[player_roles_full_name] = weights[player_roles_full_name].replace(WEIGHTS_DICT_PLAYERS)

# Calculate overall per role
player_roles = positions.loc[positions["positions"].notna(), "short_name"].unique()
players = pd.concat([players, calculate_overall(positions, weights, players, player_roles)],
                    axis="columns")

# Get the top three overalls
result = players[player_roles].apply(get_top_overalls, axis="columns")
players = pd.concat([players, result], axis="columns")

for col in players.columns:
    print(col + " - " + str(players.iloc[0,:][col]))
