.DEFAULT_GOAL := main-beholder
DATA?="${shell pwd}"
GPU=0
DOCKER_FILE=ngmot/docker/Dockerfile
DOCKER=docker
NAME?=beholder:latest
CONFIG?=cfg_beholder.yml
DEV?=0

build:
	$(DOCKER) build -t $(NAME) -f $(DOCKER_FILE) .

main-beholder: build
	$(DOCKER) run --gpus '"device=$(GPU)"' -it --rm -v $(DATA):/workspace --net=host \
	$(NAME) python3 main_beholder.py $(CONFIG)
