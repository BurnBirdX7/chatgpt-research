#!/usr/bin/env sh

# Clone colbert
git clone https://github.com/stanford-futuredata/ColBERT.git colbert

# Download and unpack ColBERTv2 checkpoint
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -O checkpoint.tar.gz
tar -xvzf checkpoint.tar.gz
mv colbertv2.0 checkpoint

# Download FEVER dataset
wget https://fever.ai/download/fever/train.jsonl -O collections/fever.jsonl
