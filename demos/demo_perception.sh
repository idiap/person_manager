#!/bin/bash

SESSION=idiap

check_session=`tmux ls |grep $SESSION`
if [ -n "$check_session" ]; then
    echo -e "\e[1m\e[31m+++ ERROR: tmux session with name $SESSION is already running. +++\e[0m"
    echo "Try:"
    echo
    echo -e "\ttmux a -t $SESSION"
    echo
    echo "to attach to it and either keep on using this session or close all the terminals (Ctrl+C and Ctrl+D) and rerun this script."
    echo
    exit
fi

tmux -2 new-session -d -s $SESSION
# Setup a window for tailing log files
tmux rename-window -t $SESSION:0 'core'
tmux new-window -t $SESSION:1 -n 'robot'
tmux new-window -t $SESSION:2 -n 'perception'

tmux select-window -t $SESSION:0
tmux split-window -v
tmux select-pane -t 0
tmux send-keys "roscore" C-m
tmux select-pane -t 1
tmux send-keys "htop" C-m
tmux select-pane -t 0

tmux select-window -t $SESSION:1
tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver naoqi_driver.launch network_interface:=enp5s0"
tmux split-window -v
tmux select-pane -t 1
tmux send-keys "rosservice call /naoqi_driver/motion/wake_up"
tmux split-window -v
tmux select-pane -t 2
tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver register_depth.launch"
tmux select-pane -t 0

tmux select-window -t $SESSION:2
tmux split-window -v
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch person_manager perception.launch naoqi:=1 tracker_delay:=50 tracker_scale:=0.5 tracker_detector:=openheadpose tracker_particles:=50 tracker_min_height:=0.12 manager_keep_threshold:=0.02 tracker_visu:=0 manager_visu:=1 with_audio:=1"
tmux select-pane -t 1
tmux send-keys "rosrun person_manager demo_stimulus_reaction.py -m 0.15"

# Set default window
tmux select-window -t $SESSION:0

# Attach to session
tmux -2 attach-session -t $SESSION
