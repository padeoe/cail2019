FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 修改源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g;s/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update

# 修改pip源
RUN mkdir ~/.pip \
    && printf '%s\n%s\n%s\n' '[global]' 'trusted-host = mirrors.aliyun.com' \
    'index-url = https://mirrors.aliyun.com/pypi/simple'>> ~/.pip/pip.conf

# 配置python环境
RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/environment
ADD requirements.txt .
RUN pip install -r requirements.txt