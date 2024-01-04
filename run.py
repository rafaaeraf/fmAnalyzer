import numpy as np
import os
import pandas as pd
import random
import sys

################## Inputs ##################
dataDir = "C:\\fmAnalyzer\\data"
outputDir = "C:\\fmAnalyzer\\output"
clubName = "Blyth"
weightsDict = {"green": 5, "blue": 3, "normal": 1}
teamFormation = ["GR", "D(E)", "D(C)", "D(C)", "D(D)", "MD", "M(C)", "M(C)", "MO(C)", "PL(C)", "PL(C)"]


################## Methods ##################
def getDfType(df):
    if ("Posição" in df.keys()):
        return "players"
    elif ("Função" in df.keys()):
        return "coaches"

def getDfTypeDetailed(df, clubName):
    if (getDfType(df) == "players"):
        # If all players are from the club or lent by the club, them this file is a club export file
        if (all((df['Clube'] == clubName) | (df['Emprestado Por'] == clubName))):
            return "myClubPlayers"
        else:
            return "searchPlayers"
    elif (getDfType(df) == "coaches"):
        if (all(df['Clube'] == clubName)):
            return "myClubCoaches"
        else:
            return "searchCoaches"
    else:
        sys.exit("ERROR: Data is not for coaches neither players")

def readData(dataDir):
    allFiles = []
    for dataFile in os.listdir(dataDir):
        fileFullPath = os.path.join(dataDir, dataFile)
        allFiles.append({"file": fileFullPath, "timestamp": os.path.getmtime(fileFullPath)})
    allFiles = pd.DataFrame(allFiles)
    allFiles = allFiles.sort_values("timestamp")

    # Create list of dataframes
    playersList = []
    coachesList = []

    for file in allFiles["file"]:
        # Read html
        tmpDf = pd.read_html(file, header = 0, encoding = "utf-8", na_values = ["-", "- -"])
        if (len(tmpDf) != 1):
            sys.exit("ERROR: All generated tables should have length 1")
        tmpDf = tmpDf[0]
        # Add relevant info to df
        tmpDf["sourceFile"] = file
        tmpDf["timestamp"] = allFiles[allFiles["file"] == file]["timestamp"].values[0]
        # Get dataframe type
        dfType = getDfType(tmpDf)
        tmpDf["dfType"] = dfType
        tmpDf["dfTypeDetailed"] = getDfTypeDetailed(tmpDf, clubName)
        # Append to the list
        if dfType == "players":
            playersList.append(tmpDf)
        else:
            coachesList.append(tmpDf)

    # Concatenate DataFrames outside the loop
    players = pd.concat(playersList, axis=0, ignore_index=True)
    coaches = pd.concat(coachesList, axis=0, ignore_index=True)
    
    return players, coaches

def getPersonClubStatus(row, clubName):
    if (row["dfTypeDetailed"] == "myClubPlayers"):
        if (row["Clube"] == clubName):
            return "myClub"
        else:
            return "myClubLent"
    elif (row["dfTypeDetailed"] == "searchPlayers"):
        if (pd.isna(row["Clube"])):
            return "hasNoClub"
        else:
            return "hasClub"
    elif (row["dfTypeDetailed"] == "myClubCoaches"):
        return "myClub"
    elif (row["dfTypeDetailed"] == "searchCoaches"):
        if (pd.isna(row["Clube"])):
            return "hasNoClub"
        else:
            return "hasClub"

def parsePosition(origPosition):
    origPosition = origPosition.split(",")
    ret = []
    for o in origPosition:
        side = None
        if "(" in o:
            side = o.split("(")[1].split(")")[0]
        positions = o.split("/")
        for p in positions:
            currPosition = p.replace(" ", "").split("(")[0]
            if side == None:
                ret.append(currPosition)
            else:
                for s in side:
                    ret.append(currPosition + "(" + s + ")")
    return ",".join(ret)

def getRelevantAttributes(editedPositions, attributes):
    if "GR" in editedPositions:
        return attributes[(attributes["gkAttribute"] == "both") | (attributes["gkAttribute"] == "gk")]["shortName"]
    else:
        return attributes[(attributes["gkAttribute"] == "both") | (attributes["gkAttribute"] == "nonGk")]["shortName"]

def checkDataCompletion(row, attributes):
    editedPositions = row["editedPositions"]
    relevantAttributes = getRelevantAttributes (editedPositions, attributes)
    filteredDf = row[relevantAttributes]
    if all(pd.isna(filteredDf.values)):
        return "allIncomplete"
    elif any(pd.isna(filteredDf.values)):
        return "incomplete"
    elif any(filteredDf.astype(str).str.contains("-")):
        return "completeButRange"
    else:
        return "complete"

def convertDfColumnsToNumeric(df, columns):
    dfCopy = df.copy()
    for col in columns:
        dfCopy[col] = pd.to_numeric(dfCopy[col], errors = "coerce")
    return dfCopy

def getAverageAttributePerPosition(players, attributes):
    allPositions = [p.split(",") for p in players["editedPositions"].unique()]
    allPositions = np.unique(np.concatenate(allPositions))
    averageAttributesPos = pd.DataFrame()
    completePlayers = players[players["dataCompletion"] == "complete"]
    for pos in allPositions:
        relevantAttributes = getRelevantAttributes(pos, attributes)
        filteredDf = completePlayers[completePlayers["editedPositions"].str.contains(pos, regex = False)]
        filteredDf = filteredDf[relevantAttributes]
        filteredDf = convertDfColumnsToNumeric(filteredDf, relevantAttributes)
        averageAttributesPos = pd.concat([averageAttributesPos, filteredDf.mean().to_frame().T], ignore_index = True)
    averageAttributesPos["positions"] = allPositions
    return averageAttributesPos

def handleMissingAttributes(row, attributes, averageAttributesPos):
    if row["dataCompletion"] in ["allIncomplete", "complete"]:
        return row
    row = row.copy()
    relevantAttributes = getRelevantAttributes(row, attributes)
    playerPosition = random.choice(row["editedPositions"].split(","))
    for attributeName in relevantAttributes:
        if pd.isna(row[attributeName]):
            selectedAttribute = averageAttributesPos.loc[averageAttributesPos["positions"] == playerPosition, attributeName]
            if len(selectedAttribute) != 1:
                sys.exit("ERROR: Unexpected length for the selectedAttribute")
            row.at[attributeName] = float(selectedAttribute.iloc[0])
        elif "-" in row[attributeName]:
            rangeAttribute = row[attributeName].split("-")
            row.at[attributeName] = (float(rangeAttribute[0]) + float(rangeAttribute[1])) / 2
    return row

# For each player role, gets the weights for that role, multiply by each player attributes
# and divide by sum of the weigths, so we get a result between 0 and 20
def calculateOverall(positions, weights, players, playerRoles):
    ret = pd.DataFrame()
    for role in playerRoles:
        roleFullName = positions.loc[positions["shortName"] == role, "fullName"].iloc[0]
        filteredWeights = weights.loc[~pd.isna(weights[roleFullName]), roleFullName]
        total = players[filteredWeights.keys()].multiply(filteredWeights, axis = "columns")
        total = total.apply(sum, axis = "columns") / filteredWeights.sum()
        ret[role] = total
    return ret

# Function to get the top three best overalls and their respective role names
# TODO: Improve this method by implementing it as a loop
# TODO: Return full name of role instead of short name
def getTopOveralls(row):
    sortedRow = row.sort_values(ascending = False)
    firstOverallCol = sortedRow.index[0]
    secondOverallCol = sortedRow.index[1]
    thirdOverallCol = sortedRow.index[2]
    firstOverall = row[firstOverallCol]
    secondOverall = row[secondOverallCol]
    thirdOverall = row[thirdOverallCol]
    if firstOverallCol.startswith("norm - "):
        columnNames = ['firstNormOverall', 'firstNormOverallRole', 'secondNormOverall', 'secondNormOverallRole', 'thirdNormOverall', 'thirdNormOverallRole']
        firstOverallCol = firstOverallCol.replace("norm - ", "")
        secondOverallCol = secondOverallCol.replace("norm - ", "")
        thirdOverallCol = thirdOverallCol.replace("norm - ", "")
    else:
        columnNames = ['firstOverall', 'firstOverallAttribute', 'secondOverall', 'secondOverallAttribute', 'thirdOverall', 'thirdOverallAttribute']
    return pd.Series([firstOverall, firstOverallCol, secondOverall, secondOverallCol, thirdOverall, thirdOverallCol], 
                     index = columnNames)

# Remove positions that a player does not play on from a dataframe
def removeUnplayedPositions(row, positions):
    # Get all positions the player does not play on
    notPlayedPositions = positions[positions["positions"].apply(
        lambda x: all(item not in str(x) for item in row["editedPositions"].split(",")) if not pd.isna(x) else False)]
    # Transform all of those positions to NA
    row[notPlayedPositions["shortName"]] = pd.NA

    return row

# Returns the best player, the role and overall value of a dataframe containing only overalls
def findMaxValueLocation(df):
    # Replace NaN values with a very low number
    dfFilled = df.fillna(-float('inf'))

    # Find the location of the maximum value
    maxLocation = dfFilled.stack().idxmax()

    # Extract idu, role, and maximum value
    idu, role = maxLocation
    maxValue = df.at[idu, role]

    return idu, role, maxValue

################## Main ##################

# Get current dir and read attributes weights csvs
currDir = os.path.dirname(os.path.abspath(__file__))
attributes = pd.read_csv(currDir + "\\relevantAttributes\\attributes.csv")
positions = pd.read_csv(currDir + "\\relevantAttributes\\positions.csv")
weights = pd.read_csv(currDir + "\\relevantAttributes\\weights.csv", index_col = "shortName")

# Read data dir, get all files and order from old to new
[players, coaches] = readData(dataDir)

# Drop duplicates considering unique ID and keeping most recent date
players = players.sort_values("timestamp").drop_duplicates(["IDU"], keep = "last")
coaches = coaches.sort_values("timestamp").drop_duplicates(["IDU"], keep = "last")
players = players.set_index("IDU")
coaches = coaches.set_index("IDU")

# Get person club status
players["clubStatus"] = players.apply(lambda x: getPersonClubStatus(x, clubName), 1)
coaches["clubStatus"] = coaches.apply(lambda x: getPersonClubStatus(x, clubName), 1)

# Parse players positions - Go from "MD, M/MO (DC), PL (C)" to MD,M(D),M(C),MO(D),MO(C),PL(C)
players["editedPositions"] = players["Posição"].apply(parsePosition, 1)

# Classify player/coach on attribute availability - complete, complete but range, incomplete, all incomplete
players["dataCompletion"] = players.apply(lambda x: checkDataCompletion(x, attributes), 1)

# Get average attribute value per position
averageAttributesPos = getAverageAttributePerPosition(players, attributes)

# Handle missing data for all playes that are "complete but range" or "incomplete"
players = players.apply(lambda x: handleMissingAttributes(x, attributes, averageAttributesPos), 1)

# Convert attributes to numeric
players = convertDfColumnsToNumeric(players, attributes.loc[attributes["playerCoach"] == "player", "shortName"])

# Replace the attribute colors with weight values
weights.replace(weightsDict, inplace=True)

# Calculate overall per role
playerRoles = positions.loc[positions["positions"].notna(), "shortName"].unique()
players = pd.concat([players, calculateOverall(positions, weights, players, playerRoles)], axis = "columns")

# Get the top three overalls
result = players[playerRoles].apply(getTopOveralls, axis = "columns")
players = pd.concat([players, result], axis = "columns")

# Calculate normalized overalls
# TODO: Extract a method
maxOveralls = players[playerRoles].apply(max, axis = "rows")
minOveralls = players[playerRoles].apply(min, axis = "rows")
ret = 100 * (players[playerRoles] - minOveralls) / (maxOveralls - minOveralls)
playerRolesNorm = "norm - " + playerRoles # Names of player roles normalized overall columns
ret.columns = playerRolesNorm
players = pd.concat([players, ret], axis = "columns")

# Get the top three normalized overalls
result = players[playerRolesNorm].apply(getTopOveralls, axis = "columns")
players = pd.concat([players, result], axis = "columns")

# Get best 11
# TODO: Extract a method
# Get only players currently on my club
filteredPlayers = players[players["clubStatus"] == "myClub"]
# Remove all overalls from positions the each player do not play on
filteredPlayers = filteredPlayers.apply(removeUnplayedPositions, axis = 1, positions = positions)
# Positions to be filled
missingPositions = teamFormation.copy()
bestEleven = []
# Loop
while len(missingPositions) > 0:
    # Get the roles relevant for team formation missing positions
    filteredPositions = positions[positions["positions"].apply(lambda x: any(item in str(x) for item in missingPositions) if not pd.isna(x) else False)]
    # Only roles relevant for current missing positions
    filteredPlayers = filteredPlayers[filteredPositions["shortName"]]
    # Get best player for all of missing positions
    idu, role, maxValue = findMaxValueLocation(filteredPlayers)
    # Get a position that the team still needs, that is compatible with the role and the player plays
    rolePositions = positions.loc[positions["shortName"] == role, "positions"].item().split(",")
    playerPositions = players.loc[idu, "editedPositions"].split(",")
    selectedPosition = random.choice(list(set(rolePositions) & set(playerPositions) & set(missingPositions)))
    # Add player to bestEleven dataframe and remove the player from possible players to be used and position from missing positions
    bestEleven.append([idu, selectedPosition, role, maxValue])
    missingPositions.remove(selectedPosition)
    filteredPlayers = filteredPlayers.drop(idu)
bestEleven = pd.DataFrame(bestEleven, columns = ["IDU", "position", "role", "overall"])
bestEleven = bestEleven.set_index("IDU")


filteredPlayers[filteredPositions["shortName"]]
filteredPlayers[filteredPositions["shortName"]]
# pega todos os roles que o cara não joga e transforma em nan

# Get best second squad

# Get best under 19

# Get best 11 lent



#weigths = convertDfColumnsToNumeric(weights, [c for c in weights.columns if c != "shortName"])

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


players.to_csv(outputDir + "\\players.csv")
players.to_excel(outputDir + "\\players.xlsx")
print("end")
