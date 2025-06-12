FROM movesrwth/stormpy:1.9.0 AS prep
LABEL authors="runge"
ARG BUILD_JOBS=4

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

#WORKDIR /
#RUN git clone https://github.com/AAU-Dat/CuPAAL.git
WORKDIR /CuPAAL
COPY . .

WORKDIR /CuPAAL/build
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$BUILD_JOBS

# run the experiments
# multistage only used to avoid rebuilding dependencies
FROM prep AS experiment

COPY $PRISM_FILE_PATH $PRISM_FILE_PATH
COPY --chmod=755 entrypoint.sh .
COPY jajapy_part.py .
ENTRYPOINT ["./entrypoint.sh"]

#ENTRYPOINT [ "./CuPAAL", "-o", "../results/calculated_model.txt", "-r", "../results/data.csv" ]
#CMD [ "-m", "../cupaal_model.txt", "-s", "../cupaal_training_set.txt" ]

#ENTRYPOINT ["sh", "-c", "ls && sleep infinity"]
#ENTRYPOINT [ "sh", "-c", "echo 'Hello from inside the container!' > /results/output.txt && sleep infinity" ]
