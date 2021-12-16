#!/bin/bash

PYTHONHOME="/vol/research/xmodal_dl/mmf/bin"
HOME="/vol/research/xmodal_dl/mmf"

echo $HOME
echo 'args:' $@

$PYTHONHOME/python $HOME/mmf_cli/run.py env.data_dir=$HOME/data env.save_dir=$HOME/save $@
