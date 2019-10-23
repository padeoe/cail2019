FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 修改源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g;s/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update

# 修改pip源
RUN pip install pip -U
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 配置python环境
RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/environment
RUN pip install Cython==0.29.7
ADD docker/requirements.txt .
RUN pip install -r requirements.txt