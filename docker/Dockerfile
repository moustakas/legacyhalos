FROM ubuntu:18.04 as builder

RUN apt-get update && apt-get install -y autoconf automake gcc g++ make gfortran wget && apt-get clean all

# Install MPI following https://docs.nersc.gov/programming/shifter/how-to-use
RUN mkdir /build/

RUN cd /build \
    && wget https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz \
    && tar xvzf mpich-3.3.tar.gz \
    && cd /build/mpich-3.3 \
    && ./configure \
    && make -j4 \
    && make install \
    && make clean

RUN echo 'PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
ENV PATH=$PATH:/opt/conda/bin

RUN wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    conda config --add channels intel && \
    conda config --set always_yes yes && \
    conda update conda && \
    conda install python=3.7 pip numpy && \
    conda clean -a -y

RUN cd /build \
    && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz \
    && tar xvzf mpi4py-3.0.3.tar.gz \
    && cd /build/mpi4py-3.0.3 \
    && python setup.py build \
    && python setup.py install

RUN rm -rf /build

#### Stage 2 ####
FROM legacysurvey/legacypipe:preburn-lslga
 
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/conda /opt/conda
 
RUN /sbin/ldconfig

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
