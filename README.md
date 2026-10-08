# edge-cloud-inference-scheduler

uvicorn inference.server:app --reload

docker build -t edge-inference .
docker images // for verification

docker run --rm -p 8000:8000 edge-inference // run the container

// run 3 containers
docker run --rm --name edge_1 -p 8001:8000 edge-inference
docker run --rm --name edge_2 -p 8002:8000 edge-inference
docker run --rm --name cloud -p 8003:8000 edge-inference
