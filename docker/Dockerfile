FROM legacysurvey/legacypipe:DR9.5.5pre

RUN for x in \
    ipython \
    pydl \
    scikit-image \
    ; do pip --no-cache-dir install $x; done

# Remove the policy.xml file so we do not get an 'exhausted cache resources'
# error when we build mosaics for very large systems.
RUN echo '<policymap></policymap>' > /etc/ImageMagick-6/policy.xml

# legacyhalos
WORKDIR /src
RUN git clone https://github.com/moustakas/legacyhalos.git legacyhalos \
    && cd legacyhalos && git describe

RUN echo 'PYTHONPATH=/src/legacyhalos/py:$PYTHONPATH'
RUN echo 'PATH=/src/legacyhalos/bin:$PATH'

# update legacyhalos
RUN cd /src/legacyhalos && git pull && git describe && echo 1

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
