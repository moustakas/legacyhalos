Build a Docker container for the legacyhalos project.
=====================================================

Build a cross-platform docker container as documented [here](https://www.docker.com/blog/faster-multi-platform-builds-dockerfile-cross-compilation-guide), [here](https://blog.jaimyn.dev/how-to-build-multi-architecture-docker-images-on-an-m1-mac/), and [here](https://docs.nersc.gov/development/shifter/how-to-use/).

First, pull the latest 
```
docker pull legacysurvey/legacypipe:DR10.1.3

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
```
and then either
```
docker buildx create --name mybuild --use
```
or
```
docker buildx use mybuild
```
and then

```
docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t legacysurvey/legacyhalos:v1.3 .
docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t legacysurvey/legacyhalos:latest .
```

To enter the container (with a shell prompt) on a laptop do:
```
docker pull legacysurvey/legacyhalos:latest
docker run -it legacysurvey/legacyhalos:latest
```

Or at NERSC:
```
shifterimg pull docker:legacysurvey/legacyhalos:v1.3
shifterimg pull docker:legacysurvey/legacyhalos:latest
shifter --image docker:legacysurvey/legacyhalos:latest bash
```
