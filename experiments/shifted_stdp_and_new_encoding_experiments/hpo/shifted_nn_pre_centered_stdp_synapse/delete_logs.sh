#!/bin/bash

# Start from current directory
ROOT_DIR="."

# JOB_ID whose logs should be deleted
JOB_ID="$1"

# Checking whether JOB_ID was passed
if [[ -z "$ROOT_DIR" || -z "$JOB_ID" ]]; then
    echo "Usage: $0 <JOB_ID>"
    exit 1
fi

# For protection against unintentional deletion of unnecessary folders:
# JOB_ID must be number (only digits)
if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: JOB_ID must contain only digits."
    exit 1
fi

# Search and Remove
find "$ROOT_DIR" -name "*$JOB_ID*" | while read -r dir_or_file; do
    echo "Deleting: $dir_or_file"
    rm -rf "$dir_or_file"
done
