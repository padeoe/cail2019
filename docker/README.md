# 使用 Docker 搭建开发环境、训练模型、测试打包
[![](https://img.shields.io/docker/cloud/automated/padeoe/cail)](https://hub.docker.com/r/padeoe/cail "Docker Cloud Automated build")

## Set up development environment
This will help you build a simulation environment that is consistent with the online system  of the competition, and provide an environment that is easy to develop, including installing all dependencies and providing ssh acess and jupyter notebook.

**1.get `dev` image**

pull image from dockerhub
```bash
docker pull padeoe/cail:dev
```
or build it by yourself
```bash
docker build \
    -t cail:dev \
    --build-arg apt_mirror="mirrors.aliyun.com" \
    --build-arg pypi_mirror="https://mirrors.aliyun.com/pypi/simple" \
    -f docker/dev/Dockerfile .
```

**2.run container**

```bash
BERT_PRETRAINED_MODEL_DIR="/data/resources/bert"
SSH_PORT=2229
JUPYTER_PORT=2230
docker run \
    --name cail_dev \
    --restart always \
    --runtime nvidia \
    -v $BERT_PRETRAINED_MODEL_DIR:/bert \
    -p $SSH_PORT:22 \
    -p $JUPYTER_PORT:8888 \
    -d \
    padeoe/cail:dev
```

Now we can access ssh at `ssh -p 2229 root@localhost` and jupyter notebook at [http://localhost:2230](http://localhost:2230).

The ssh password is `cail` by default, you can change it in [Dockerfile](dev/Dockerfile)

## Train models
This container will perform model training and output model files.

**1.get `train` image**

pull image from dockerhub
```bash
docker pull padeoe/cail:train
```
or build it by yourself
```bash
docker build \
    -t cail:train \
    --build-arg apt_mirror="mirrors.aliyun.com" \
    --build-arg pypi_mirror="https://mirrors.aliyun.com/pypi/simple" \
    -f docker/train/Dockerfile .
```

**2.prepare data**

The dataset provided by organizer is downloaded during image building process,
so the dataset is bundled in docker image in `/cail/data`.
If you want to use your own dataset, put dataset files at `DATA_DIR`, 
the layout should be like this:

```console
$ tree data/
data/
├── raw
│   └── CAIL2019-SCM-big
│       └── input.txt
└── test
    ├── ground_truth.txt
    └── input.txt

3 directories, 3 files
```

**3.train models**

```bash
BERT_PRETRAINED_MODEL="/data/resources/bert/pytorch_chinese_L-12_H-768_A-12/"
OUTPUT_MODELS_DIR="$PWD/model"
DATA_DIR="/data/padeoe/project/cail/data"
docker run \
    --name cail-train \
    --rm \
    -it \
    --runtime nvidia \
    -e LOCAL_USER_ID=`id -u $USER` \
    -v $BERT_PRETRAINED_MODEL:/bert/pytorch_chinese_L-12_H-768_A-12 \
    -v ${DATA_DIR}:/cail/data \
    -v ${OUTPUT_MODELS_DIR}:/cail/model \
    padeoe/cail:train
```
The model will be saved at `OUTPUT_MODELS_DIR`.

## Test and submission
This container will test your model and compress the codes into zip format required by the organizer.

**1.get `submit` image**

pull image from dockerhub
```bash
docker pull padeoe/cail:submit
```
or build it by yourself
```bash
docker build \
    -t cail:submit \
    --build-arg apt_mirror="mirrors.aliyun.com" \
    --build-arg pypi_mirror="https://mirrors.aliyun.com/pypi/simple" \
    -f docker/submit/Dockerfile .
```

**2.run test and zip codes**

```bash
docker run \
    --rm \
    -it \
    --runtime nvidia \
    -e LOCAL_USER_ID=`id -u $USER` \
    -e SUBMIT_FILES="main.py model.py model" \
    -v "$PWD":/codes \
    -v "$PWD/data/submit_zip":/output_zip \
    padeoe/cail:submit
```
It will execute `main.py` and `judger.py` and output the the accuracy of evaluation.
Finally, the compressed codes and models for submit will stored in `$PWD/data/submit_zip`.
