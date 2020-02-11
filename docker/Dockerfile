FROM legacysurvey/legacypipe:dr9e-pre1

RUN mkdir /home/legacyhalos
ENV HOME /home/legacyhalos
RUN for x in \
    /root/.astropy \
    /root/.ipython \
    /root/.config \
    /root/.qdo \
    /root/.bashrc \
    ; do cp -R $x /home/legacyhalos/; done

RUN for x in \
    ipython \
    pydl \
    ; do pip --no-cache-dir install $x; done

LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
