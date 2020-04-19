FROM python:3.7.7

WORKDIR /root/
RUN git clone -b multiprocess https://github.com/eyounx/ZOOpt.git && \
cd ZOOpt && \
python setup.py build && \
python setup.py install 

RUN pip install pandas torch==1.2.0

RUN pip install psutil
#COPY . /root/ncov

ENTRYPOINT ["/bin/bash"]
