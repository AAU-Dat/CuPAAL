FROM ubuntu:24.04

ARG NUMBER_OF_CORES=4

RUN apt update && apt upgrade -y
RUN apt install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev -y


##### BUILD
WORKDIR /opt

RUN git clone -b stable https://github.com/moves-rwth/storm.git /opt/storm

RUN mkdir /opt/storm/build
WORKDIR /opt/storm/build

RUN cmake ..
RUN make install

#RUN make -j${NUMBER_OF_CORES}