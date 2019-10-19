FROM cail:base

# install zip
RUN apt-get -y install zip

# install gosu
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

# entrypoint to execute command, fix permission problem
ADD docker/submit/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod a+x /docker-entrypoint.sh

# 用于映射到主机上用户代码目录
ENV CODES_USER_DIR '/codes'
# 需要提交的代码文件列表
ENV SUBMIT_FILES 'main.py'
# 将 `CODES_USER_DIR` 目录中的 `SUBMIT_FILES` 文件拷贝并存储的路径
ENV CODES_COPY_DIR '/codes_copy'
# 存储最终提交的代码压缩包的目录
ENV CODES_ZIP_DIR '/output_zip'

RUN mkdir $CODES_USER_DIR /output
WORKDIR $CODES_COPY_DIR
# script for test
COPY judger.py .
# script for running test and compressing files
COPY docker/submit/submit.sh .
RUN chmod a+x submit.sh
# ground truth and test samples
COPY data/test/ground_truth.txt /data/ground_truth.txt
COPY data/test/input.txt /input/input.txt

# run test and zip files for submmit
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["./submit.sh"]