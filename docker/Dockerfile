FROM legacysurvey/legacypipe:dr9e3

RUN for x in \
    ipython \
    pydl \
    ; do pip --no-cache-dir install $x; done

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
