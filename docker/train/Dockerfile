FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 修改apt源
ARG apt_mirror="archive.ubuntu.com"
RUN sed -i "s/archive.ubuntu.com/$apt_mirror/g;s/security.ubuntu.com/$apt_mirror/g" /etc/apt/sources.list
RUN apt-get update

# 修改pypi源
ARG pypi_mirror="https://pypi.python.org/simple"
RUN pip config set global.index-url $pypi_mirror

# 安装gosu
RUN gpg --no-tty --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# 修改locale
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# 修改时区
RUN apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# 安装python依赖
RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/environment
RUN pip install pip -U
ADD requirements.txt .
RUN pip install -r requirements.txt

# 安装nvidia apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /cail
ADD . .

# download data
RUN python data.py

COPY docker/train/docker-entrypoint.sh  /
RUN chmod a+x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "train.py"]