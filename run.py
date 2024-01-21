import os
import random
import sys

import numpy as np
import pandas as pd


################## Inputs ##################
DATA_DIR = "C:\\fmAnalyzer\\data"
OUTPUT_DIR = "C:\\fmAnalyzer\\output"
CLUB_NAME = "Blyth"
WEIGHTS_DICT = {"green": 5, "blue": 3, "normal": 1}
TEAM_FORMATION = ["GR", "D(E)", "D(C)", "D(C)", "D(D)", "MD", "M(C)", "M(C)", "MO(C)", "PL(C)",
                   "PL(C)"]

################## Methods ##################
def get_df_type(df):
    if "Posição" in df.keys():
        return "players_internal"
    elif "Função" in df.keys():
        return "coaches"

def get_df_type_detailed(df):
    if get_df_type(df) == "players_internal":
        # If all players_internal are from the club or lent by the club, then this file is a
        # club export file
        if all((df['Clube'] == CLUB_NAME) | (df['Emprestado Por'] == CLUB_NAME)):
            return "my_club_players"
        else:
            return "search_players"
    elif get_df_type(df) == "coaches":
        if all(df['Clube'] == CLUB_NAME):
            return "my_club_coaches"
        else:
            return "search_coaches"
    else:
        sys.exit("ERROR: Data is not for coaches neither players_internal")

def read_data():
    all_files = []
    for data_file in os.listdir(DATA_DIR):
        file_full_path = os.path.join(DATA_DIR, data_file)
        all_files.append({"file": file_full_path, "timestamp": os.path.getmtime(file_full_path)})
    all_files_df = pd.DataFrame(all_files)
    all_files_df = all_files_df.sort_values("timestamp")

    # Create list of dataframes
    players_list = []
    coaches_list = []

    for file in all_files_df["file"]:
        # Read html
        tmp_df = pd.read_html(file, header=0, encoding="utf-8", na_values=["-", "- -"])
        if len(tmp_df) != 1:
            sys.exit("ERROR: All generated tables should have length 1")
        tmp_df = tmp_df[0]
        # Add relevant info to df
        tmp_df["source_file"] = file
        tmp_df["timestamp"] = all_files_df[all_files_df["file"] == file]["timestamp"].values[0]
        # Get dataframe type
        df_type = get_df_type(tmp_df)
        tmp_df["df_type"] = df_type
        tmp_df["df_type_detailed"] = get_df_type_detailed(tmp_df)
        # Append to the list
        if df_type == "players_internal":
            players_list.append(tmp_df)
        else:
            coaches_list.append(tmp_df)

    # Concatenate DataFrames outside the loop
    players_ret = pd.concat(players_list, axis=0, ignore_index=True)
    coaches_ret = pd.concat(coaches_list, axis=0, ignore_index=True)

    return players_ret, coaches_ret

def get_person_club_status(row):
    if row["df_type_detailed"] == "my_club_players":
        if row["Clube"] == CLUB_NAME:
            return "my_club"
        else:
            return "my_club_lent"
    elif row["df_type_detailed"] == "search_players":
        if pd.isna(row["Clube"]):
            return "has_no_club"
        else:
            return "has_club"
    elif row["df_type_detailed"] == "my_club_coaches":
        return "my_club"
    elif row["df_type_detailed"] == "search_coaches":
        if pd.isna(row["Clube"]):
            return "has_no_club"
        else:
            return "has_club"

def parse_position(orig_position):
    orig_position = orig_position.split(",")
    ret = []
    for o in orig_position:
        side = None
        if "(" in o:
            side = o.split("(")[1].split(")")[0]
        positions_internal = o.split("/")
        for p in positions_internal:
            curr_position = p.replace(" ", "").split("(")[0]
            if side is None:
                ret.append(curr_position)
            else:
                for s in side:
                    ret.append(curr_position + "(" + s + ")")
    return ",".join(ret)

def get_relevant_attributes(edited_positions, attributes_internal):
    if "GR" in edited_positions:
        return attributes_internal[(attributes_internal["gk_attribute"] == "both") |
                                   (attributes_internal["gk_attribute"] == "gk")]["short_name"]
    else:
        return attributes_internal[(attributes_internal["gk_attribute"] == "both") |
                                   (attributes_internal["gk_attribute"] == "non_gk")]["short_name"]

def check_data_completion(row, attributes_internal):
    edited_positions = row["edited_positions"]
    relevant_attributes = get_relevant_attributes(edited_positions, attributes_internal)
    filtered_df = row[relevant_attributes]
    if all(pd.isna(filtered_df.values)):
        return "all_incomplete"
    elif any(pd.isna(filtered_df.values)):
        return "incomplete"
    elif any(filtered_df.astype(str).str.contains("-")):
        return "completeButRange"
    else:
        return "complete"

def convert_df_columns_to_numeric(df, columns):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy

def get_all_possible_positions(players_internal):
    all_positions = [p.split(",") for p in players_internal["edited_positions"].unique()]
    all_positions = np.unique(np.concatenate(all_positions))
    return all_positions

def get_average_attribute_per_position(players_internal, attributes_internal):
    all_positions = get_all_possible_positions(players_internal)
    ret = pd.DataFrame()
    complete_players = players_internal[players_internal["data_completion"] == "complete"]
    for pos in all_positions:
        relevant_attributes = get_relevant_attributes(pos, attributes_internal)
        filtered_df = complete_players[complete_players["edited_positions"]
                                       .str.contains(pos,regex=False)]
        filtered_df = filtered_df[relevant_attributes]
        filtered_df = convert_df_columns_to_numeric(filtered_df, relevant_attributes)
        ret = pd.concat([ret, filtered_df.mean().to_frame().T], ignore_index=True)
    ret["positions"] = all_positions
    return ret

def handle_missing_attributes(row, attributes_internal, average_attributes_pos_internal):
    if row["data_completion"] in ["all_incomplete", "complete"]:
        return row
    row = row.copy()
    relevant_attributes = get_relevant_attributes(row, attributes_internal)
    player_position = random.choice(row["edited_positions"].split(","))
    for attribute_name in relevant_attributes:
        if pd.isna(row[attribute_name]):
            selected_attribute = average_attributes_pos_internal.loc[
                average_attributes_pos_internal["positions"] == player_position, attribute_name]
            if len(selected_attribute) != 1:
                sys.exit("ERROR: Unexpected length for the selected_attribute")
            row.at[attribute_name] = float(selected_attribute.iloc[0])
        elif "-" in row[attribute_name]:
            range_attribute = row[attribute_name].split("-")
            row.at[attribute_name] = (float(range_attribute[0]) + float(range_attribute[1])) / 2
    return row

# For each player role, gets the weights_internal for that role, multiply by each player
# attributes_internal and divide by sum of the weigths, so we get a result between 0 and 20
def calculate_overall(positions_internal, weights_internal, players_internal,
                      player_roles_internal):
    ret = pd.DataFrame()
    for role in player_roles_internal:
        role_full_name = positions_internal.loc[
            positions_internal["short_name"] == role, "full_name"].iloc[0]
        filtered_weights = weights_internal.loc[
            ~pd.isna(weights_internal[role_full_name]), role_full_name]
        total = players_internal[filtered_weights.keys()].multiply(filtered_weights, axis="columns")
        total = total.apply(sum, axis="columns") / filtered_weights.sum()
        ret[role] = total
    return ret

# Function to get the top three best overalls and their respective role names
# TODO: Improve this method by implementing it as a loop
# TODO: Return full name of role instead of short name
def get_top_overalls(row):
    sorted_row = row.sort_values(ascending=False)
    first_overall_col = sorted_row.index[0]
    second_overall_col = sorted_row.index[1]
    third_overall_col = sorted_row.index[2]
    first_overall = row[first_overall_col]
    second_overall = row[second_overall_col]
    third_overall = row[third_overall_col]
    if first_overall_col.startswith("norm - "):
        column_names = ['first_norm_overall', 'first_norm_ovverall_role', 'second_norm_overall',
                        'second_norm_overall_role', 'third_norm_overall', 'third_norm_overall_role']
        first_overall_col = first_overall_col.replace("norm - ", "")
        second_overall_col = second_overall_col.replace("norm - ", "")
        third_overall_col = third_overall_col.replace("norm - ", "")
    else:
        column_names = ['first_overall', 'first_overall_attribute', 'second_overall',
                        'second_overall_attribute', 'third_overall', 'third_overall_attribute']
    return pd.Series([first_overall, first_overall_col, second_overall, second_overall_col,
                      third_overall, third_overall_col], index=column_names)

# Given all overalls, get the normalized value going from 0 to 100 for each role
def calculate_normalized_overall(players_internal, player_roles_internal):
    max_overalls = players_internal[player_roles_internal].apply(max, axis="rows")
    min_overalls = players_internal[player_roles_internal].apply(min, axis="rows")
    ret = 100 * (players_internal[player_roles_internal] - min_overalls) / (max_overalls -
                                                                             min_overalls)
    # Names of player roles normalized overall columns
    player_roles_norm = "norm - " + player_roles_internal
    ret.columns = player_roles_norm
    return ret

# Remove positions_internal that a player does not play on from a dataframe
def remove_unplayed_positions(row, positions_internal):
    # Get all positions_internal the player does not play on
    not_played_positions = positions_internal[positions_internal["positions"].apply(
        lambda x: all(item not in str(x) for
                       item in row["edited_positions"].split(",")) if not pd.isna(x) else False)]
    # Transform all of those positions_internal to NA
    row[not_played_positions["short_name"]] = pd.NA
    return row

# Returns the best player, the role and overall value of a dataframe containing only overalls
def find_max_value_location(df):
    # Replace NaN values with a very low number
    df_filled = df.fillna(-float('inf'))

    # Find the location of the maximum value
    max_location = df_filled.stack().idxmax()
    # Extract idu, role, and maximum value
    idu, role = max_location
    max_value = df.at[idu, role]
    return idu, role, max_value

# Get a clean dataframe removing players_internal from a list of dataframes
def exclude_players(players_internal, players_to_exclude):
    filtered_players = players_internal.copy()
    # If players_internal to exclude is provided, remove these players_internal
    if players_to_exclude is not None:
        for df in players_to_exclude:
            players_to_drop = [index for index in df.index if index in filtered_players.index]
            filtered_players = filtered_players.drop(players_to_drop)
    return filtered_players

# Given a players_internal dataframe returns the best players_internal
# If team formation is provided, 11 players_internal for these positions_internal are returned
def get_top_players(players_internal, positions_internal, player_roles_internal,
                    players_to_exclude=None, team_formation_internal=None, age=None, num_players=5,
                    reference_team=None, knowledge=None):
    filtered_players = exclude_players(players_internal, players_to_exclude)
    # Filter by age
    if age is not None:
        filtered_players = filtered_players[filtered_players["Idade"] <= age]
    # Filter by knowledge
    if knowledge is not None:
        filtered_players = filtered_players[filtered_players["Nív. Conh."] <= knowledge]

    if isinstance(reference_team, pd.DataFrame):
        # If comparing to a team, list all positions_internal we need to find
        # players_internal better than the list
        missing_positions = list(set(reference_team["position"]))
    elif team_formation_internal is not None:
        # If mounting a team, get all positions_internal we need to fill
        missing_positions = team_formation_internal.copy()
    # If position based, remove all overalls from positions_internal the each player do not play on
    if isinstance(reference_team, pd.DataFrame) or team_formation_internal is not None:
        filtered_players = filtered_players.apply(remove_unplayed_positions, axis=1,
                                                  positions_internal=positions_internal)
    # Get overalls
    filtered_players = filtered_players[player_roles_internal]
    ret = []
    add_player_to_ret = True

    while True:
        if team_formation_internal is not None or isinstance(reference_team, pd.DataFrame):
            # Get the roles relevant for team formation missing positions_internal
            filtered_positions = positions_internal[positions_internal["positions"].apply(
                lambda x: any(item in str(x) for
                              item in missing_positions) if not pd.isna(x) else False)]
            # Only roles relevant for current missing positions_internal
            filtered_players = filtered_players[filtered_positions["short_name"]]

            if filtered_players.shape[0] == 0:
                break

        # Get best player for all of missing positions_internal
        idu, role, max_value = find_max_value_location(filtered_players)

        # If position based, check in which position player will play
        if team_formation_internal is not None or isinstance(reference_team, pd.DataFrame):
            # Get a position that the team still needs, that is compatible with the role
            # and the player plays
            role_positions = positions_internal.loc[
                positions_internal["short_name"] == role, "positions"].item().split(",")
            player_positions = players_internal.loc[idu, "edited_positions"].split(",")
            acceptable_positions = list(set(role_positions) & set(player_positions) &
                                        set(missing_positions))
            # In some cases, player can play in one role that fills a position he plays, but do not
            # play for a position we need
            if len(acceptable_positions) > 0:
                selected_position = random.choice(acceptable_positions)
            else:
                # Give up on this player
                filtered_players = filtered_players.drop(idu)
                continue
        else:
            selected_position = players_internal.loc[idu, "edited_positions"]

        if isinstance(reference_team, pd.DataFrame):
            # If we are comparing to a team, now it is time to decide if we should add this best
            # player to the list or not
            # If player is better than the reference team, add to ret
            if max_value > min(reference_team.loc[reference_team["position"] == selected_position,
                                                  "overall"]):
                add_player_to_ret = True
            else:
                # Case our best player is not good enough for that position, remove the position
                # from the list as we are giving up on it
                add_player_to_ret = False
                missing_positions.remove(selected_position)

        if add_player_to_ret:
            # Add player to ret dataframe and remove the player from possible players_internal
            # to be used
            ret.append([idu, selected_position, role, max_value])
            filtered_players = filtered_players.drop(idu)

        if team_formation_internal is not None:
            # If mounting a team, remove position from from missing positions_internal
            missing_positions.remove(selected_position)

        # Should we leave?
        if team_formation_internal is not None or isinstance(reference_team, pd.DataFrame):
            if len(missing_positions) == 0:
                break
        else:
            num_players -= 1
            if num_players == 0:
                break

    ret = pd.DataFrame(ret, columns=["IDU", "position", "role", "overall"])
    ret = ret.set_index("IDU")
    return ret

# Given a set of filtered dataframes and tags pairs, returns a column of the original original
# dataframe containing these tags for the relevant row contained in the filtered dataframes
def get_analysis_status(players_internal, df_and_tag):
    players_copy = pd.DataFrame(index=players_internal.index, columns=["analysis_status"])

    for df, tag in df_and_tag:
        players_copy.loc[players_copy.index.isin(df.index), "analysis_status"] = tag

    return players_copy["analysis_status"]

# Function to convert percentage strings to float
def handle_percentage_values(percentage_str):
    # Remove the percentage sign and convert to float
    return float(percentage_str.strip("%")) / 100


################## Main ##################
def main():
    # Get current directory and read attributes weights CSVs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    attributes = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "attributes.csv"))
    positions = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "positions.csv"))
    weights = pd.read_csv(os.path.join(curr_dir, "relevant_attributes", "weights.csv"),
                          index_col="short_name")

    # Read data directory, get all files, and order from old to new
    players, coaches = read_data()

    # Drop duplicates considering unique ID and keeping most recent date
    players = players.sort_values("timestamp").drop_duplicates(["IDU"], keep="last")
    coaches = coaches.sort_values("timestamp").drop_duplicates(["IDU"], keep="last")
    players = players.set_index("IDU")
    coaches = coaches.set_index("IDU")

    # Handle percentage values
    players["Nív. Conh."] = players["Nív. Conh."].apply(handle_percentage_values)

    # Get person club status
    players["club_status"] = players.apply(get_person_club_status, axis=1)
    coaches["club_status"] = coaches.apply(get_person_club_status, axis=1)

    # Parse players positions - Go from "MD, M/MO (DC), PL (C)" to MD,M(D),M(C),MO(D),MO(C),PL(C)
    players["edited_positions"] = players["Posição"].apply(parse_position)

    # Classify player/coach on attribute availability - complete, complete but range,
    # incomplete, all incomplete
    players["data_completion"] = players.apply(lambda x: check_data_completion(x, attributes),
                                               axis=1)

    # Get average attribute value per position
    average_attributes_pos = get_average_attribute_per_position(players, attributes)

    # Handle missing data for all players that are "complete but range" or "incomplete"
    players = players.apply(lambda x: handle_missing_attributes(x, attributes,
                                                                average_attributes_pos), axis=1)

    # Convert attributes to numeric
    players = convert_df_columns_to_numeric(
        players, attributes.loc[attributes["player_coach"] == "player", "short_name"])

    # Replace the attribute colors with weight values
    weights.replace(WEIGHTS_DICT, inplace=True)

    # Calculate overall per role
    player_roles = positions.loc[positions["positions"].notna(), "short_name"].unique()
    players = pd.concat([players, calculate_overall(positions, weights, players, player_roles)],
                        axis="columns")

    # Get the top three overalls
    result = players[player_roles].apply(get_top_overalls, axis="columns")
    players = pd.concat([players, result], axis="columns")

    # Calculate normalized overalls
    players = pd.concat([players, calculate_normalized_overall(players, player_roles)],
                        axis="columns")

    # Get the top three normalized overalls
    result = players["norm - " + player_roles].apply(get_top_overalls, axis="columns")
    players = pd.concat([players, result], axis="columns")

    # Get best 11 and best 2nd team
    best_eleven = get_top_players(players[players["club_status"] == "my_club"], positions,
                                  player_roles, team_formation_internal=TEAM_FORMATION)
    best_2nd_team = get_top_players(players[players["club_status"] == "my_club"], positions,
                                    player_roles, [best_eleven], TEAM_FORMATION)
    # Get lent players good enough for best eleven and best 2nd team
    lent_best_eleven = get_top_players(players[players["club_status"] == "my_club_lent"],
                                       positions, player_roles, reference_team=best_eleven)
    lent_best_2nd_team = get_top_players(players[players["club_status"] == "my_club_lent"],
                                         positions, player_roles, [lent_best_eleven],
                                         reference_team=best_2nd_team)
    # Get best 11 under 19 and 17
    under_19_best_eleven = get_top_players(
        players[players["club_status"].isin(["my_club_lent", "my_club"])], positions, player_roles,
        [best_eleven, best_2nd_team, lent_best_eleven, lent_best_2nd_team],
        team_formation_internal=TEAM_FORMATION, age=19)
    under_17_best_eleven = get_top_players(
        players[players["club_status"].isin(["my_club_lent", "my_club"])], positions, player_roles,
        [best_eleven, best_2nd_team, lent_best_eleven, lent_best_2nd_team, under_19_best_eleven],
        team_formation_internal=TEAM_FORMATION, age=17)
    # Get the bad players
    bad_players = exclude_players(players[players["club_status"] == "my_club"],
                                  [best_eleven, best_2nd_team, under_19_best_eleven,
                                   under_17_best_eleven])
    # Get best lent players
    bad_lent = exclude_players(players[players["club_status"] == "my_club_lent"],
                               [lent_best_eleven, lent_best_2nd_team, under_19_best_eleven,
                                under_17_best_eleven])

    # Handle players with low knowledge
    no_knowledge = players[players["data_completion"] == "all_incomplete"]
    low_knowledge_best_2nd_team = get_top_players(
        players[players["df_type_detailed"] == "search_players"], positions, player_roles,
        [no_knowledge], reference_team=best_2nd_team, knowledge=0.3)
    low_knowledge_best_2nd_team_under_19 = get_top_players(
        players[players["df_type_detailed"] == "search_players"], positions, player_roles,
        [no_knowledge, low_knowledge_best_2nd_team], reference_team=under_19_best_eleven,
        age=19, knowledge=0.3)
    low_knowledge_best_top_5 = get_top_players(
        players[players["df_type_detailed"] == "search_players"], positions, player_roles,
        [no_knowledge, low_knowledge_best_2nd_team, low_knowledge_best_2nd_team_under_19],
        knowledge=0.3)
    low_knowledge_best_top_5_under_19 = get_top_players(
        players[players["df_type_detailed"] == "search_players"], positions, player_roles,
        [no_knowledge, low_knowledge_best_2nd_team, low_knowledge_best_2nd_team_under_19,
         low_knowledge_best_top_5], age=19, knowledge=0.3)
    low_knowledge_bad_players = exclude_players(
        players[(players["df_type_detailed"] == "search_players") & (players["Nív. Conh."] <= 0.3)],
        [no_knowledge, low_knowledge_best_2nd_team, low_knowledge_best_2nd_team_under_19,
         low_knowledge_best_top_5, low_knowledge_best_top_5_under_19])
    low_knowledge_suggest_scout = exclude_players(
        players[(players["df_type_detailed"] == "search_players") & (players["Nív. Conh."] <= 0.3)],
        [low_knowledge_bad_players])
    # Players with knowledge
    hire_best_eleven = get_top_players(players[players["df_type_detailed"] == "search_players"],
                                       positions, player_roles,
                                       [no_knowledge, low_knowledge_bad_players,
                                        low_knowledge_suggest_scout], reference_team=best_eleven)
    hire_best_2nd_team = get_top_players(players[players["df_type_detailed"] == "search_players"],
                                         positions, player_roles,
                                         [no_knowledge, low_knowledge_bad_players,
                                          low_knowledge_suggest_scout, hire_best_eleven],
                                          reference_team=best_2nd_team)
    hire_best_under_19 = get_top_players(players[players["df_type_detailed"] == "search_players"],
                                         positions, player_roles,
                                         [no_knowledge, low_knowledge_bad_players,
                                          low_knowledge_suggest_scout, hire_best_eleven,
                                          hire_best_2nd_team],
                                          reference_team=under_19_best_eleven, age=19)
    non_team_formation_positions = list(set(get_all_possible_positions(players)) -
                                        set(TEAM_FORMATION))
    best_other_positions = get_top_players(players[players["df_type_detailed"] == "search_players"],
                                           positions, player_roles,
                                           [no_knowledge, low_knowledge_bad_players,
                                            low_knowledge_suggest_scout, hire_best_eleven,
                                            hire_best_2nd_team, hire_best_under_19],
                                            team_formation_internal=non_team_formation_positions)
    bad_hire_players = exclude_players(players[players["df_type_detailed"] == "search_players"],
                                       [no_knowledge, low_knowledge_bad_players,
                                        low_knowledge_suggest_scout, hire_best_eleven,
                                        hire_best_2nd_team, hire_best_under_19,
                                        best_other_positions])

    # Add a tags column to the players dataframe
    players["analysis_status"] = get_analysis_status(
        players, [(best_eleven, "best_11"), (best_2nd_team, "best_2nd_team"),
                  (lent_best_eleven, "lentbest_11"), (lent_best_2nd_team, "lent_best_2nd_team"),
                  (under_19_best_eleven, "under_19_best_11"),
                  (under_17_best_eleven, "under_17_best_11"), (bad_players, "bad_players"),
                  (bad_lent, "bad_lent"), (no_knowledge, "no_knowledge"),
                  (low_knowledge_bad_players, "low_knowledge_bad_players"),
                  (low_knowledge_suggest_scout, "low_knowledge_suggest_scout"),
                  (hire_best_eleven, "hirebest_11"), (hire_best_2nd_team, "hire_best_2nd_team"),
                  (hire_best_under_19, "hire_best_under_19"),
                  (best_other_positions, "best_other_positions"),
                  (bad_hire_players, "bad_hire_players")])

    # Save to CSV and Excel
    players.to_csv(os.path.join(OUTPUT_DIR, "players.csv"))
    players.to_excel(os.path.join(OUTPUT_DIR, "players.xlsx"))

    print("end")

if __name__ == "__main__":
    main()

#weigths = convertDfColumnsToNumeric(weights, [c for c in weights.columns if c != "short_name"])

#i = 0
#for a in players['Pas']:
#    if type(a) is np.ndarray:
#        print(i)
#    i = i + 1

# check unique content per column
#for key in players.keys():
#    print(key)
#    print(len(players[key].unique()))
#    print(players[key].unique())
#    print("\n\n\n\n\n\n\n______________________________________")
#for key in coaches.keys():
#    print(key)
#    print(len(coaches[key].unique()))
#    print(coaches[key].unique())
#    print("\n\n\n\n\n\n\n______________________________________")
