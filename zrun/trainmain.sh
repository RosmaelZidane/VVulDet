# Train the main model without domain knowledge
cd ..
echo "Training without domain data"
echo "   The process takes time. "
echo "The Node2vec should process all functions and generate contextualized graph embedding."
# Sleep for 30 seconds
sleep 30
python3 ./sourcescripts/mainmodel.py

python3 ./sourcescripts/testtop-x.py