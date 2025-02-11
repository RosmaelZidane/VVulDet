
import numpy as np
import os

import utils.utills as imp 
import utils.nodeedgesdata as xtra
import utils.preprocessdata as prep

# SETUP
NUM_JOBS = 1# 1
JOB_ARRAY_NUMBER = 0 

# Read Data
df = prep.CVEFixes()

df = df.iloc[::-1]
#df = df.transpose
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.CVEFixes()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = imp.get_dir(imp.processed_dir() / row["dataset"] / "before")
    savedir_after = imp.get_dir(imp.processed_dir() / row["dataset"] / "after")
    
    # Add the directory where to save code descriptions into txt file
    savedir_description_CVE = imp.get_dir(imp.processed_dir() / row['dataset'] / "CVEdescription")
    savedir_description_CWE = imp.get_dir(imp.processed_dir() / row['dataset'] / "CWEdescription")
    savedir_sample_func = imp.get_dir(imp.processed_dir() / row['dataset'] / "CWE_Samples")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.py"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.py"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])
            
    # add code to write vulnerability descriptions that will be used as domain informations       
    fpath3 = savedir_description_CVE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath3}.txt") :# and len(row['CVE_vuldescription']) > 5:
        with open(fpath3, 'w') as f:
            f.write(row['CVE_vuldescription'])       
    # add code to write description from the Mitre 1000 of the CWE
    fpath4 = savedir_description_CWE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath4}.txt") : # and len(row['CWE_vuldescription'])>4:
        with open(fpath4, 'w') as f:
            f.write(row['CWE_vuldescription'])
    # add the code to get sample from the mitre save as text file as well
    fpath5 = savedir_sample_func / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath5}.txt"): # and len(row["CWE_Sample"])> 5: # 8307/8326
        with open(fpath5, 'w') as f:
            f.write(row['CWE_Sample'])
    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        xtra.full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        xtra.full_run_joern(fpath2, verbose=3)
    
        

if __name__ == "__main__":
    imp.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)
