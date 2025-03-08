# This file enables data preprocessing, processing, 
# and running Joern to parse functions for obtaining node and edge data.

#!/bin/bash
cd ..

# Run the first Python script
python3 ./sourcescripts/getmetadata.py

# Print a message to the console
echo "All the metadata is being processed"

echo "Starting Joern..."
echo "    Starting Joern..."
echo "        Starting Joern..."
# Sleep for 30 seconds
sleep 30

# Run the second Python script
python3 ./sourcescripts/getgraphdata.py
