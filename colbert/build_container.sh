#!/usr/bin/env sh

if [ ! -f "colbertv2.0.tar.gz" ]; then
  wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
fi
sudo docker build -t colbert .

if [ $? -ne 0 ]; then
  echo "Build FAILED"
  exit 1
fi

echo "Build successful!"

BASE_CONTAINER=$(sudo docker run -dit --gpus all colbert bash)

if [ $? -ne 0 ]; then
  echo "Couldn't run the container!"
  exit 2
fi

echo "Running the container... ID = $BASE_CONTAINER"

# Prepare data
sudo docker exec -it "$BASE_CONTAINER" bash --login -c "python -m colbert prepare_fever" &&
sudo docker exec -it "$BASE_CONTAINER" bash --login -c "python -m colbert create_index fever"

SERVER_IMAGE=$(sudo docker commit "$BASE_CONTAINER")
SERVER_CONTAINER=$(sudo docker run -dit --gpus all "$SERVER_IMAGE" bash --login -c "python -m colbert fever_server")

# Commit
sudo docker commit "$SERVER_CONTAINER" colbert-fever > /dev/null

sudo docker stop "$BASE_CONTAINER" > /dev/null
sudo docker stop "$SERVER_CONTAINER" > /dev/null

echo "Done!"
echo "Run the server with: docker run --gpus all colbert-fever"


