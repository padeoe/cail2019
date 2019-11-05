FROM padeoe/cail:base

# 开启ssh
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
RUN pip install pip -U
RUN pip install prompt-toolkit jupyter
#RUN mkdir -p ~/.jupyter/custom && \
#    echo ".CodeMirror pre, .CodeMirror-dialog, .CodeMirror-dialog .CodeMirror-search-field, .terminal-app .terminal {\n \tfont-family: Consolas;\n \tfont-size: 12pt;\n}" > ~/.jupyter/custom/custom.css
RUN pip install jupyterthemes
RUN jt -t chesterish -f source

## matplotlib 安装中文字体
RUN apt-get -y install fontconfig unzip
RUN wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansSC.zip -O /tmp/NotoSansSC.zip && unzip /tmp/NotoSansSC.zip -d /usr/share/fonts/truetype/NotoSansSC
RUN fc-cache -f -v

## 安装nvidia apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 78c38db467c378759b4278a4ca8547763da5c91d && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /cail

EXPOSE 22 8888

ENTRYPOINT service ssh start && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root