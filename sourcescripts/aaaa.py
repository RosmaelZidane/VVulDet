import utils.utills as imp     


print(imp.project_dir())
print(imp.storage_dir())
print(imp.outputs_dir())





import pandas as pd      

datapath = "/home/wp3/VVulDet/sourcescripts/storage/external"
df = pd.read_csv(f"{datapath}/domain_CVEFixes-Python1.csv")

n = 50
dfv = df[df['vul']==1]
dfv = dfv.sample(n)
dfnv = df[df['vul']==0]
dfnv = dfnv.sample(n)

ddf = pd.concat([dfv, dfnv], ignore_index=True)
ddf.to_csv(f"{datapath}/domain_CVEFixes-Python.csv", index = False)