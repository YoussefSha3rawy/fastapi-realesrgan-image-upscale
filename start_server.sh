#!/bin/bash

git clone https://github.com/xinntao/Real-ESRGAN.git

# install python dependencies
python -m venv venv
source venv/bin/activate && pip install -r requirements.txt && cd Real-ESRGAN &&
 python setup.py develop && cd .. && cp -r Real-ESRGAN/realesrgan . &&
 rm -rf Real-ESRGAN && uvicorn main:app --reload --host 0.0.0.0 --port 8000