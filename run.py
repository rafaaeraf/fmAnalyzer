import numpy as np
import os
import pandas as pd
import sys

# Inputs
dataDir = "C:\\fmAnalyzer\\data"
clubName = "Blyth"
greenWeight = 5
blueWeight = 3
normalWeight = 1


# Methods
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

def getPersonClubStatus(row, clubName):
    if (row["dfTypeDetailed"] == "myClubPlayers"):
        if (row["Clube"] != clubName):
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
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    return df



# Current dir
currDir = os.path.dirname(os.path.abspath(__file__))

# Read data dir, get all files and order from old to new
allFiles = []
for dataFile in os.listdir(dataDir):
    fileFullPath = os.path.join(dataDir, dataFile)
    allFiles.append({"file": fileFullPath, "timestamp": os.path.getmtime(fileFullPath)})
allFiles = pd.DataFrame(allFiles)
allFiles = allFiles.sort_values("timestamp")

# Create result dataframes
players = pd.DataFrame()
coaches = pd.DataFrame()

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
    # Append to the dataframe
    if (dfType == "players"):
       players = pd.concat([players, tmpDf], axis=0, ignore_index=True)
    else:
        coaches = pd.concat([coaches, tmpDf], axis=0, ignore_index=True)

# Drop duplicates considering unique ID and keeping most recent date
players = players.sort_values("timestamp").drop_duplicates(["IDU"], keep = "last")
coaches = coaches.sort_values("timestamp").drop_duplicates(["IDU"], keep = "last")

# Get person club status
players["clubStatus"] = players.apply(lambda x: getPersonClubStatus(x, clubName), 1)
coaches["clubStatus"] = coaches.apply(lambda x: getPersonClubStatus(x, clubName), 1)

# Parse players positions - Go from "MD, M/MO (DC), PL (C)" to MD,M(D),M(C),MO(D),MO(C),PL(C)
players["editedPositions"] = players["Posição"].apply(parsePosition, 1)

# Read attribute weights information
attributes = pd.read_csv(currDir + "\\relevantAttributes\\attributes.csv")
positions = pd.read_csv(currDir + "\\relevantAttributes\\positions.csv")
weights = pd.read_csv(currDir + "\\relevantAttributes\\weights.csv")

# Classify player/coach on attribute availability - complete, complete but range, incomplete, all incomplete
players["dataCompletion"] = players.apply(lambda x: checkDataCompletion(x, attributes), 1)

# Get average attribute value per position
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

# Handle missing data for all playes that are complete but range or incomplete
# calculate when possible 





for key in players.keys():
    print(key)
    print(len(players[key].unique()))
    print(players[key].unique())
    print("\n\n\n\n\n\n\n______________________________________")

for key in coaches.keys():
    print(key)
    print(len(coaches[key].unique()))
    print(coaches[key].unique())
    print("\n\n\n\n\n\n\n______________________________________")


print("bla")
