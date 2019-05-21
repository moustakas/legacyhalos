Build a Docker container for the legacyhalos project.
=====================================================

Building or updating the container
----------------------------------

```
docker build . -t flagnarg/legacyhalos --file Dockerfile-legacyhalos
docker push flagnarg/legacyhalos:latest
```

To enter the container (with a shell prompt) on a laptop do:
```
docker pull flagnarg/legacyhalos:latest
docker run -it flagnarg/legacyhalos:latest
```

Or at NERSC:
```
shifterimg pull docker:flagnarg/legacyhalos:latest
shifter --image docker:flagnarg/legacyhalos:latest bash
```

Adding to `legacypipe`
----------------------

```
docker build . -t flagnarg/legacypipe:latest --file Dockerfile-legacypipe
docker push flagnarg/legacypipe:latest
```

To enter the container on a laptop and at NERSC, respectively, do
```
docker pull flagnarg/legacypipe:latest
docker run -it flagnarg/legacypipe:latest
```

```
shifterimg pull docker:flagnarg/legacypipe:latest
shifter --image docker:flagnarg/legacypipe:latest bash
```