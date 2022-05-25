#!/bin/bash

PYTHONHOME="/vol/research/xmodal_dl/mmf-env/bin"
HOME="/vol/research/xmodal_dl/mmf"
TIME=`date "+%Y-%m-%d-%H-%M-%S"`

echo $HOME
echo 'args:' $@

cd $HOME/
$PYTHONHOME/wandb login edc6324e6001d34f00aa0088ffbec6ff29180f2d
$PYTHONHOME/python mmf_cli/run.py $@
