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

# Extra steps for mpi4py on NERSC
RUN apt-get update        && \
    apt-get upgrade --yes && \
    apt-get install --yes    \
        libmpich-dev  \
        python3-dev && \
    apt-get clean all

RUN python3 -m pip install --no-binary=mpi4py mpi4py

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
