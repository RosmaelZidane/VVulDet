# Get Joern and the python dataset
# the csv is stored in a google file store at 
# https://drive.google.com/file/d/14vtngKXaBPI43aKRfd6-PoV3peDtd5XU/view?usp=sharing
# cd ..

if [[ -d sourcescripts/storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents sourcescripts/storage/external
fi

cd sourcescripts/storage/external

if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi

if [[ ! -f "domain_CVEFixes-Python.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=14vtngKXaBPI43aKRfd6-PoV3peDtd5XU
    unzip domain_CVEFixes-Python.zip
    rm domain_CVEFixes-Python.zip
else
    echo "Already downloaded Python version of the CVEFixes data"
fi