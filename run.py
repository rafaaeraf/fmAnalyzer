import numpy as np
import os
import pandas as pd
import sys

# Inputs
dataDir = "C:\\fmAnalyzer\\data"
clubName = "Blyth"


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

# TODO: Calculate overall for all positions
read attributes tables
handle missing data - create a status for each row - complete, complete but range, incomplete, all incomplete
calculate when possible 





print("bla")
