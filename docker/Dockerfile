FROM legacysurvey/legacypipe:DR10.1.3

RUN for x in \
    ipython \
    pydl \
    pyyaml \
    h5py \
    colossus \
    gnureadline \
    corner \
    scikit-image \
    ; do pip --no-cache-dir install $x; done

# Extra steps for mpi4py at NERSC: build mpich then install mpi4py
RUN apt-get update        && \
    apt-get upgrade --yes && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        libcurl4             \
        wget             &&  \
    apt-get clean all    &&  \
    rm -rf /var/lib/apt/lists/*

ARG mpich=4.0.2
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch ./configure            && \
    make -j 16                                                              && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

RUN python3 -m pip install mpi4py

# Remove the policy.xml file so we do not get an 'exhausted cache resources'
# error when we build mosaics for very large systems.
RUN echo '<policymap></policymap>' > /etc/ImageMagick-6/policy.xml

ENV IPYTHONDIR /tmp/ipython-config

# legacyhalos
WORKDIR /src
RUN git clone https://github.com/moustakas/legacyhalos.git legacyhalos \
    && cd legacyhalos && git describe

ENV PYTHONPATH=/src/legacyhalos/py:$PYTHONPATH
ENV PATH=/src/legacyhalos/bin:$PATH

# update legacyhalos
RUN cd /src/legacyhalos && git pull && git describe && echo 4

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
