DATA?="${HOME}"
APP?=$(shell pwd)

build-gpu:
	nvidia-docker build --build-arg TF_VERSION=1.9.0-gpu-py3 -t keras_gpu .

build-cpu:
	docker build --build-arg TF_VERSION=1.9.0-py3 -t keras_cpu .

notebook: build-gpu
	nvidia-docker run -it -v $(APP):/app -v $(DATA):/data --rm --net=host --env KERAS_BACKEND=tensorflow keras_gpu

bash-cpu: build-cpu
	docker run -it -v $(APP):/app -v $(DATA):/data --rm keras_cpu bash

bash-gpu: build-gpu
	nvidia-docker run -it -v $(APP):/app -v $(DATA):/data --rm keras_gpu bash
