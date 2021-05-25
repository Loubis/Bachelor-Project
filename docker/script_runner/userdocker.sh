sudo userdocker run -i -t --cpus=6 --cpuset-cpus="19-24" -v '/usr/bin/sudo:/usr/local/bin/sudo:ro' -v '/usr/lib/sudo:/usr/lib/sudo:ro' -v '/mnt/nfs/mi_backup/datashare/docker/sudoers:/etc/sudoers:ro' tensorflow/tensorflow:2.3.0-gpu bash


sudo apt-get update && sudo apt-get install ffmpeg graphviz && pip install tqdm spleeter pydot
