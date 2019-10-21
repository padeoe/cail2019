#!/bin/bash

cd $CODES_USER_DIR
cp -r $SUBMIT_FILES $CODES_COPY_DIR

cd $CODES_COPY_DIR
python3 main.py
python3 judger.py
zip -FSr $CODES_ZIP_DIR/cail_$(date +"%Y_%m_%d").zip $SUBMIT_FILES
