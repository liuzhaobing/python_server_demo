FROM python:3.9
COPY . /opt

WORKDIR /opt

RUN pip3 install -r requirements.txt -i "https://pypi.tuna.tsinghua.edu.cn/simple"

ENTRYPOINT ["/bin/bash", "run.sh"]