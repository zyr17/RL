autossh -i "~/.ssh/id_rsa" -M 38383 -N -R 0.0.0.0:33070:localhost:6096 -R 0.0.0.0:6099:localhost:6099 ubuntu@t.zyr17.cn
