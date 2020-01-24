FROM ubuntu:18.04 as builder

RUN apt-get update && apt-get install -y autoconf automake gcc g++ make gfortran wget && apt-get clean all

# # Install MPI following https://docs.nersc.gov/programming/shifter/how-to-use
# RUN mkdir /build/
# RUN cd /build \
#     && wget https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz \
#     && tar xvzf mpich-3.3.tar.gz \
#     && cd /build/mpich-3.3 \
#     && ./configure \
#     && make -j4 \
#     && make install \
#     && make clean
# 
# RUN /sbin/ldconfig

#### Stage 2 ####
FROM legacysurvey/legacypipe:preburn-lslga
 
RUN mkdir -p /build/
 
COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/bin /usr/bin
 
# LABEL Maintainer="John Moustakas jmoustakas@siena.edu"
# 
# RUN pip --no-cache-dir install mpi4py
# 
# # Add the development version of photutils
# # RUN pip3 uninstall -y photutils \
# #     && cd /build \
# #     && git clone https://github.com/astropy/photutils.git \
# #     && cd photutils \
# #     && python3 setup.py install
# 
# RUN rm -rf /build
# 
# # # Clone legacyhalos
# # RUN git clone http://github.com/moustakas/legacyhalos.git legacyhalos && echo 1
# # ENV PYTHONPATH $PYTHONPATH:/src/legacyhalos/py
