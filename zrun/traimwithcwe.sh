# cd ..
cd sourcescripts/storage/outputs/
rm -rf checkpoints
cd ..
cd cache
rm -rf codebert_method_level/
rm -rf CVEFixes_linevd_codebert_pdg+raw/
rm -rf codebert_method_level/

cd ..
cd ..
cd ..

python3 ./sourcescripts/model_cwe.py