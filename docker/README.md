Build a Docker container for the legacyhalos project.
=====================================================

This uses the Intel compilers, which introduces two complications:

- you need to access the license server at NERSC to build the intel compilers
- cannot be distributed (i.e., posted to Docker Hub), so NERSC provides a
- "two-stage build" recipe where you build in a full container, and then copy
- your results into a container with just the freely-distributable runtime
- components of the Intel suite.

First, add a stanza called `intel-license` to your `~/.ssh/config` file:

```
Host intel-license
Hostname cori.nersc.gov
GatewayPorts yes
LocalForward 28519 intel.licenses.nersc.gov:28519
IdentityFile ~/.ssh/nersc
IdentitiesOnly yes
```

Second, run `sshproxy` if you haven't already done so today.

Third, tunnel to NERSC's license server and then build the container (assumed
here to be on OSX):

```
ssh -fN intel-license
LOCAL_IP=$(ipconfig getifaddr $(route get nersc.gov | grep 'interface:' | awk '{print $NF}'))
docker build  --add-host intel.licenses.nersc.gov:${LOCAL_IP} -t flagnarg/legacyhalos:latest --file Dockerfile-legacyhalos . 

docker push flagnarg/legacyhalos:latest
```



```
docker build . -t flagnarg/legacyhalos --file Dockerfile-legacyhalos
docker push flagnarg/legacyhalos:latest
```

More details
------------

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
