import pandas as pd

data = pd.read_csv("playersAndAwardsMapped.csv")

mvpData = data[data["award"].str.lower() == "nba mvp"].copy()

mvpData["share"] = pd.to_numeric(mvpData["share"], errors="coerce")

mvpData = mvpData.sort_values(
    ["season", "share"],
    ascending=[True, False]
)

topThree = mvpData.groupby("season").head(3).copy()

topThree["mvp"] = topThree["winner"].astype(int)

finalData = topThree[
    [
        "season",
        "player_award",
        "g",
        "mp_per_game",
        "fg_percent",
        "x3p_percent",
        "ft_percent",
        "trb_per_game",
        "ast_per_game",
        "stl_per_game",
        "blk_per_game",
        "pts_per_game",
        "mvp"
    ]
].copy()

finalData.columns = [
    "season",
    "player",
    "games",
    "minutesPerGame",
    "fieldGoalPercentage",
    "threePointPercentage",
    "freeThrowPercentage",
    "reboundsPerGame",
    "assistsPerGame",
    "stealsPerGame",
    "blocksPerGame",
    "pointsPerGame",
    "mvp"
]

features = [
    "games",
    "minutesPerGame",
    "fieldGoalPercentage",
    "threePointPercentage",
    "freeThrowPercentage",
    "reboundsPerGame",
    "assistsPerGame",
    "stealsPerGame",
    "blocksPerGame",
    "pointsPerGame"
]

for feature in features:
    finalData[feature] = pd.to_numeric(
        finalData[feature],
        errors="coerce"
    )

finalData = finalData.dropna().reset_index(drop=True)

trainingData = finalData[
    (finalData["season"] >= 2001) &
    (finalData["season"] <= 2016)
].copy()

validationData = finalData[
    (finalData["season"] >= 2017) &
    (finalData["season"] <= 2020)
].copy()

predictionData = finalData[
    finalData["season"] == 2021
].copy()

trainingData.to_csv("NBA_MVP_Training.csv", index=False)
validationData.to_csv("NBA_MVP_Validation.csv", index=False)
predictionData.to_csv("NBA_MVP_2021.csv", index=False)

print("Training data:", trainingData.shape)
print("Validation data:", validationData.shape)
print("2021 prediction data:", predictionData.shape)

print("\nTraining Data")
print(trainingData)

print("\nValidation Data")
print(validationData)

print("\n2021 Prediction Data")
print(predictionData)