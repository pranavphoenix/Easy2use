create a docker container from image

```bash
sudo docker run -it -v /path/to/folder/where/container/to/be/creaated:/workspace/ --gpus '"device=3"' --shm-size 128gb <image-name>`.
```
```bash
sudo docker run -it -v /path/to/folder/where/container/to/be/creaated:/workspace/ --gpus '"device=3"' --shm-size 128gb --network host <image-name>
```
Run jupyter notebook in docker
```bash
jupyter-lab --no-browser --port <PORT no.> --ip=0.0.0.0 --allow-root --NotebookApp.token="<token>" --NotebookApp.password="<password>"
```
load docker container
```bash
docker load -i <file.tar>
```
remember the ssh password
```bash
ssh-keygen -t rsa -b 4096
```
```bash
ssh-copy-id user@ip
```
Check which GPU a docker container is using
```bash
docker inspect <container_id> --format='{{.HostConfig.DeviceRequests}}'
```
Check the size of each subfolder in a folder
```bash
du -h -d1 /path/to/folder/
```



                       
