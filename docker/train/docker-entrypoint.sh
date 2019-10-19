#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}

chown -R $USER_ID /cail/model

useradd --shell /bin/bash -u $USER_ID -o -c "" -m user
usermod -a -G root user
export HOME=/home/user

exec /usr/local/bin/gosu user "$@"