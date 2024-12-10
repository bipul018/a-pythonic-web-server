pythonpath = echo $(which python)
echo $pythonpath
nix-shell --command 'virtualenv --python="${pythonpath}" ./python-venv; pip install mediapipe'
