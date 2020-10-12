#!/bin/bash

config="config_cpm.ini"
setup="pepper" # pepper zr300 webcam

color=
depth=
depth_config=

######################################################################

while getopts ":hc:s:" opt
do
  case ${opt} in
    c) config=${OPTARG} ;;
    s) setup=${OPTARG} ;;
    \?) echo "invalid option" -${OPTARG} ; exit 1 ;;
  esac
done

shift $(( OPTIND - 1 ))

######################################################################

echo "setup <${setup}>"
echo "config <${config}>"

if [[ "${setup}" = "pepper" ]]; then
  color="/naoqi_driver_node/camera/front/image_color"
  depth="/naoqi_driver_node/camera/depth_registered/image_rect"
  depth_config="/naoqi_driver_node/camera/depth_registered/camera_info"
elif [[ "${setup}" = "zr300" ]]; then
  color="/camera/rgb/image_rect_color"
  depth="/camera/depth_registered/sw_registered/image_rect_raw"
  depth_config="/camera/depth_registered/sw_registered/camera_info"
elif [[ "${setup}" = "webcam" ]]; then
  color="/cv_camera/image_raw"
else
  echo "Unknown setup"
  exit 1
fi

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

# Camera/Robot
if [[ "${setup}" = "pepper" ]]; then
  tmux new-window -t ${SESSION} -n "driver"
  tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver naoqi_driver.launch"
  tmux new-window -t ${SESSION} -n "register"
  tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver register_depth.launch"
elif [[ "${setup}" = "zr300" ]]; then
  tmux new-window -t ${SESSION} -n "camera"
  tmux send-keys "roslaunch realsense_camera zr300_nodelet_rgdb.launch"
  sleep 3
elif [[ "${setup}" = "webcam" ]]; then
  tmux new-window -t ${SESSION} -n "camera"
  tmux send-keys "rosrun cv_camera cv_camera_node"
  sleep 3
fi

# Tracker
tmux new-window -t ${SESSION} -n "tracker"
tmux send-keys "roslaunch particle_person_tracker main.launch color:=${color} config:=$(catkin_find particle_person_tracker ${config})"

# When depth available
if [[ ! "${setup}" = "webcam" ]]; then
  # Person manager
  tmux new-window -t ${SESSION} -n "manager"
  tmux send-keys "rosrun person_manager person_manager_node.py ${color} ${depth} ${depth_config}"

  # Visualisation
  tmux new-window -t ${SESSION} -n "visu"
  tmux send-keys "rosrun person_manager visualisation_node.py"

  # Visualisation
  tmux new-window -t ${SESSION} -n "rviz"
  tmux send-keys "rosrun rviz rviz -d $(rospack find person_manager)/rviz/pepper.rviz"
fi

# Attach to session
tmux -2 attach-session -t ${SESSION}
