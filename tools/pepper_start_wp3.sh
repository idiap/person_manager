#!/bin/bash

config="config_cv.ini"
setup="pepper" # pepper zr300 webcam

color="/naoqi_driver_node/camera/front/image_color"
depth="/naoqi_driver_node/camera/depth_registered/image_rect"
depth_config="/naoqi_driver_node/camera/depth_registered/camera_info"


SESSION=${USER}
echo "session <${SESSION}>"

check_session=`tmux ls | grep $SESSION`
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

# Run roscore
tmux -2 new-session -d -s ${SESSION}
tmux rename-window -t ${SESSION} "roscore"
tmux select-window -t ${SESSION}:roscore
tmux send-keys "roscore" C-m
sleep 2

# NQOqi driver
tmux new-window -t ${SESSION} -n "driver"
tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver naoqi_driver.launch"

# Depth registration
tmux new-window -t ${SESSION} -n "register"
tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver register_depth.launch"

# Tracker
tmux new-window -t ${SESSION} -n "tracker"
tmux send-keys "roslaunch particle_person_tracker main.launch color:=${color} config:=$(catkin_find particle_person_tracker ${config})"

# Person manager
tmux new-window -t ${SESSION} -n "manager"
tmux send-keys "rosrun person_manager person_manager_node.py ${color} ${depth} ${depth_config}"

# Visualisation
tmux new-window -t ${SESSION} -n "visu"
tmux send-keys "rosrun person_manager visualisation_node.py"

# Visualisation
tmux new-window -t ${SESSION} -n "rviz"
tmux send-keys "rosrun rviz rviz -d $(rospack find person_manager)/rviz/pepper.rviz"

# WP3 signals
tmux new-window -t ${SESSION} -n "wp3"
tmux send-keys "DISPLAY=:0 rosrun wp3_ssp engagement"

# Attach to session
tmux select-window -t $SESSION:roscore
tmux -2 attach-session -t ${SESSION}
