FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# 修改apt源
ARG apt_mirror="archive.ubuntu.com"
RUN sed -i "s/archive.ubuntu.com/$apt_mirror/g;s/security.ubuntu.com/$apt_mirror/g" /etc/apt/sources.list
RUN apt-get update

# 修改pip源
ARG pypi_mirror="https://pypi.python.org/simple"
RUN pip config set global.index-url $pypi_mirror

# 安装ssh
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:cail' | chpasswd
RUN sed -r -i 's/^\s*#?\s*PermitRootLogin\s*\S*\s*/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

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

# 配置python环境
RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/environment
RUN pip install pip -U
RUN pip install Cython==0.29.7
## 安装jupyter notebook及主题
RUN pip install prompt-toolkit jupyter
RUN pip install jupyterthemes
RUN jt -t chesterish -f source
## 安装requirements
ADD docker/requirements.txt .
RUN pip install -r requirements.txt

## matplotlib 安装中文字体
RUN apt-get -y install fontconfig unzip
RUN wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansSC.zip -O /tmp/NotoSansSC.zip && unzip /tmp/NotoSansSC.zip -d /usr/share/fonts/truetype/NotoSansSC
RUN fc-cache -f -v

## 安装nvidia apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /cail

EXPOSE 22 8888

ENTRYPOINT service ssh start && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root