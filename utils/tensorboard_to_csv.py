import pandas as pd
from tbparse import SummaryReader
from glob import glob

log_dirs = glob("/Users/hadwoslol/code/bachelors/results/tb_logs/exp1/*")
dataframe = pd.DataFrame()

for log_dir in log_dirs:
    reader = SummaryReader(log_dir)
    dfs = reader.scalars
    hparams = reader.hparams
    value = dfs.loc[3]
    value = value.drop('step')
    dataframe = dataframe.append(value)
    param = hparams.loc[2]
    dataframe = dataframe.append(param)
dataframe.to_csv("train_loss.csv")
print(dataframe)