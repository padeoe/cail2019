FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 修改源
RUN sed -i 's/archive.ubuntu.com/mirrors.nju.edu.cn/g;s/security.ubuntu.com/mirrors.nju.edu.cn/g' /etc/apt/sources.list
RUN apt-get update

# 安装gosu
RUN gpg --no-tty --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# 修改时区
RUN apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/environment

# 安装nvidia apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /cail

# 安装python依赖
ADD requirements.txt .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

ADD . .

COPY docker/train/docker-entrypoint.sh  /
RUN chmod a+x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "model.py"]