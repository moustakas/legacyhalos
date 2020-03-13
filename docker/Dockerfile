FROM legacysurvey/legacypipe:DR9.1

RUN for x in \
    ipython \
    pydl \
    ; do pip --no-cache-dir install $x; done

# Legacypipe
RUN git clone http://github.com/moustakas/legacyhalos.git legacyhalos \
    && cd legacyhalos && git describe

ENV PYTHONPATH PYTHONPATH:/src/legacyhalos/py
ENV PATH PATH:/src/legacyhalos/bin

RUN python -O -m compileall /src/legacyhalos/py/legacyhalos && \
    python -m compileall /src/legacyhalos/py/legacyhalos

# update legacyhalos
RUN cd /usr/legacyhalos && git pull && git describe && echo 0 && \
    python    -m compileall /src/legacyhalos/py/legacyhalos && \
    python -O -m compileall /src/legacyhalos/py/legacyhalos

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
