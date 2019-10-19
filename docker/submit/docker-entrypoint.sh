#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}
CODES_ZIP_DIR=${CODES_ZIP_DIR:-"/submit"}
if [[ ! -e $CODES_ZIP_DIR ]]; then
    mkdir $CODES_ZIP_DIR
fi
chown -R $USER_ID $CODES_ZIP_DIR
chown -R $USER_ID $CODES_COPY_DIR
chown -R $USER_ID /output

echo "Starting with UID : $USER_ID"
useradd --shell /bin/bash -u $USER_ID -o -c "" -m user
usermod -a -G root user
export HOME=/home/user

exec /usr/local/bin/gosu user "$@"
