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

echo "Ready to train with CVE and CWE descriptions"
echo "   The process takes time. "
echo "The Node2vec should process all functions and generate contextualized graph embedding."
# Sleep for 30 seconds
sleep 30
python3 ./sourcescripts/model_cve_cwe.py