import re


years = [2021]


for year in years:
    path = "texas_"+str(year)+"_season_stats.txt"

    with open(path) as f:
        reader = f.readlines()
        
    with open("texas_"+str(year)+"_season_stats_new.txt", "w") as n:
        for line in reader:
            line = re.sub("\s+",",",line)
            line = re.sub(",$","",line)
            n.write(line)
            n.write("\n")