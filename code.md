create a docker container from image

```bash
sudo docker run -it -v </path/to/folder/where/container/to/be/creaated>:/workspace/ --gpus '"device=<3>"' --shm-size <128>gb <image-name>`.
```
```bash
sudo docker run -it -v </path/to/folder/where/container/to/be/creaated>:/workspace/ --gpus '"device=<3>"' --shm-size <128>gb --network host <image-name>
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
ssh-copy-id <user@ip>
```
Check which GPU a docker container is using
```bash
docker inspect <container_id> --format='{{.HostConfig.DeviceRequests}}'
```
Check the size of each subfolder in a folder
```bash
du -h -d1 </path/to/folder/>
```
Run a code in a single node using bash file.sh
```bash
# !/bin/bash
set -x
torchrun \
    --nnodes=1 \
    --nproc_per_node=<NUM_GPUS> \
    --node_rank=0 \
<file.py + args> 
```
Run a code in multiple nodes using sbatch file.sh
```bash
#!/bin/bash
#SBATCH --job-name=<Job name>
#SBATCH --output=<output_log.log>         
#SBATCH --nodes=<no. of nodes>
#SBATCH --ntasks-per-node=<gpu per node>                      
#SBATCH --gres=gpu:<gpu per node>
#SBATCH --exclusive                  


# Activate virtual environment if needed
# source ~/.bashrc
# source /path/to/venv/bin/activate



# Torchrun command to launch distributed job
torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    your_script.py --your_args value
```



                       
