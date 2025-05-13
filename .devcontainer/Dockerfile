FROM movesrwth/stormpy:1.9.0 AS prep
LABEL authors="runge"
ARG BUILD_JOBS=8

RUN pip install numpy==1.26.0 && \
    pip install scipy==1.11.2 && \
    pip install alive-progress==3.1.4 && \
    pip install sympy==1.12 && \
    pip install matplotlib==3.8.1 && \
    pip install pandas==2.2.3

WORKDIR /opt
RUN git clone https://github.com/Rapfff/jajapy.git

WORKDIR /opt/jajapy
RUN python setup.py build_ext -j $BUILD_JOBS develop