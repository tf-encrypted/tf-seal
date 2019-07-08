FROM tensorflow/tensorflow:custom-op

RUN wget https://github.com/microsoft/SEAL/archive/master.zip
RUN unzip master.zip
RUN cd SEAL-master/native/src && cmake . && make && sudo make install