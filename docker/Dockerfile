FROM legacysurvey/legacypipe:DR9.1

RUN for x in \
    ipython \
    pydl \
    asdf \
    ; do pip --no-cache-dir install $x; done

# legacyhalos
WORKDIR /src
RUN git clone http://github.com/moustakas/legacyhalos.git legacyhalos \
    && cd legacyhalos && git describe

RUN echo 'PYTHONPATH=/src/legacyhalos/py:$PYTHONPATH'
RUN echo 'PATH=/src/legacyhalos/bin:$PATH'

# update legacyhalos
RUN cd /src/legacyhalos && git pull && git describe && echo 1

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
