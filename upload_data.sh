#!/bin/bash

# Simple data upload script for GPU server
# Usage: ./upload_data.sh user@server /path/to/experiment/

if [ $# -ne 2 ]; then
    echo "Usage: $0 <server> <experiment_path>"
    echo "Example: $0 user@myserver.com /home/username/experiment/"
    exit 1
fi

SERVER=$1
EXPERIMENT_PATH=$2

echo "🚀 Uploading data to GPU server..."
echo "Server: $SERVER"
echo "Path: $EXPERIMENT_PATH"
echo ""

# Create data directory on server
echo "📁 Creating data directory on server..."
ssh "$SERVER" "mkdir -p $EXPERIMENT_PATH/data/"

# Upload data files
echo "📤 Uploading data files..."
if [ -d "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" ]; then
    scp -r /Users/atharvnaphade/Downloads/atharv/deepseek-qwen/ "$SERVER:$EXPERIMENT_PATH/data/"
    echo "✅ Data uploaded successfully!"
else
    echo "❌ Local data directory not found: /Users/atharvnaphade/Downloads/atharv/deepseek-qwen"
    echo "Please update the path in this script to point to your data directory"
    exit 1
fi

echo ""
echo "🎉 Data upload completed!"
echo "You can now run the experiment on the server:"
echo "ssh $SERVER"
echo "cd $EXPERIMENT_PATH"
echo "./run_scale_experiment_server.sh"
