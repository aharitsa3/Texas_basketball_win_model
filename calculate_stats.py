import pandas as pd

df1 = pd.read_csv("team1.csv")
df2 = pd.read_csv("team2.csv")

diff_df = df1.mean() - df2.mean()

fgp = diff_df["FG%"]
fp3 = diff_df["3P%"]
fta = diff_df["FTA"]
ftp = diff_df["FT%"]
rpg = diff_df["TRB"]
apg = diff_df["AST"]
tov = diff_df["TOV"]

print(fgp, fp3, fta, ftp, rpg, apg, tov)