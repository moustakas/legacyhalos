Build a Docker container for the legacyhalos project.
=====================================================

```
docker pull legacysurvey/legacypipe:DR9.1
docker build . -t flagnarg/legacyhalos
docker push flagnarg/legacyhalos:latest

docker tag flagnarg/legacyhalos:latest flagnarg/legacyhalos:v0.0.7
docker push flagnarg/legacyhalos:v0.0.7
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
