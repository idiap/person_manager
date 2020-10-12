#!/bin/bash

SESSION="camera"

W=640
H=360
Wd=${W}
Hd=${H}
fps=15 

check_session=`tmux ls | grep $SESSION`

if [ -n "$check_session" ]; then
    echo -e "ERROR: tmux session with name $SESSION is already running."
    exit 1
fi

tmux -2 new-session -d -s ${SESSION}
tmux select-window -t ${SESSION}:0


tmux split-window -v
tmux select-pane -t 0
tmux send-keys "source ~/.bashrc && until rostopic list 2> /dev/null; do echo roscore not found; sleep 1; done && roslaunch realsense2_camera rs_camera.launch enable_sync:=1 align_depth:=1 enable_color:=1 enable_depth:=1 color_fps:=${fps} color_height:=${H} color_width:=${W} depth_fps:=${fps} depth_height:=${Hd} depth_width:=${Wd} infra_fps:=${fps} infra_height:=${Hd} infra_width:=${Wd}" C-m

tmux select-pane -t 1
tmux send-keys "python ~/monitor_intel_node.py" C-m

echo "Session ${SESSION} launched"
