#!/bin/bash

PYTHONHOME="/vol/research/xmodal_dl/mmf-env/bin"
HOME="/vol/research/xmodal_dl/mmf"
TIME=`date "+%Y-%m-%d-%H-%M-%S"`

echo $HOME
echo 'args:' $@

$PYTHONHOME/wandb login edc6324e6001d34f00aa0088ffbec6ff29180f2d
$PYTHONHOME/python $HOME/mmf_cli/run.py env.data_dir=$HOME/data env.save_dir=$HOME/save/$TIME $@
