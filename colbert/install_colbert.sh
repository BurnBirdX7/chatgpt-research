#!/usr/bin/env sh

# Clone colbert
if [ ! -d "colbert" ]; then
  echo "Cloning colbert"
  git clone https://github.com/stanford-futuredata/ColBERT.git colbert
else
  echo "Skipped ColBERT download... Directory already exists..."
fi

# Download and unpack ColBERTv2 checkpoint
if [ ! -d "checkpoint" ]; then
  echo "Downloading checkpoint"
  wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -O checkpoint.tar.gz
  tar -xvzf checkpoint.tar.gz
  mv colbertv2.0 checkpoint
  rm checkpoint.tar.gz
else
  echo "Skipped checkpoint download... Directory already exists..."
fi

# Download FEVER dataset
if [ ! -f "collections/fever.jsonl" ]; then
  echo "Downloading FEVER"

  if [ ! -d "collections" ]; then
    mkdir "collections"
  fi

  wget https://fever.ai/download/fever/train.jsonl -O collections/fever.jsonl
else
  echo "Skipped FEVER download... File already exists..."
fi
