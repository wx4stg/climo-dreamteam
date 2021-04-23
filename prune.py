import pandas as pd
pd.options.mode.chained_assignment = None

filename = "asos-2021-full"
badStations = ["CRH", "GVX", "BQX", "BBF", "MZG", "MIU", "OPM", "GYF", "25T", "VAF", "GUL", "EMK", "HQI", "HHV", "VKY"]
pandasData = pd.read_csv(str(filename+".csv"), keep_default_na=False)
for badStation in badStations:
    pandasData = pandasData[pandasData.station != badStation]
pandasData.to_csv(str(filename+".csv"), index=False)
