#!/bin/bash

# Configuration file can be:
# - config_cpm.ini : Convolutional pose machine (GPU)
# - config_cv.ini  : OpenCV face detector (less accurate)
config="config_cv.ini"

# Input device can be:
# - pepper : Robot
# - zr300  : Intel camera
# - d400   : Intel camera D400 series
# - kinect : Microsoft camera (v2)
# - webcam : Default from cv_camera
setup="pepper"

# Topic names depend on input device
color=
depth=
depth_config=

height=240
width=320
particules=50
delay=200
keep_threshold=0.25
reid_threshold=0.4

dlib_file="shape_predictor_68_face_landmarks.dat"
dlib_model=

audio_file="rec_stft_resnetctx_c25a0_1task.model.pkl"
audio_model=

######################################################################

usage()
{
  echo ""
  echo "  USAGE"
  echo ""
  echo " -c : Configuration - 'config_openpose.ini' for GPU, 'config_cv.ini' for CPU (default ${config})"
  echo " -s : Setup - pepper, zr300, kinect, webcam, d400 (default ${setup})"
  echo " -h : Height of image for the visual tracker (default ${height})"
  echo " -w : Width of image for the visual tracker (default ${width})"
  echo " -p : Number of particules for the visual tracker (default ${particules})"
  echo " -d : Delay for the visual tracker (default ${delay})"
  echo " -f : Path for the dlib model (default '${dlib_model}', if empty, looks for '${dlib_file}')"
  echo " -a : Audio model for the audio model (default '${audio_model}', if empty, looks for '${audio_file}')"
  echo ""
  exit 1
}

######################################################################

while getopts ":hc:s:f:a:" opt
do
  case ${opt} in
    c) config=${OPTARG} ;;
    s) setup=${OPTARG} ;;
    h) height=${OPTARG} ;;
    w) width=${OPTARG} ;;
    p) particules=${OPTARG} ;;
    d) delay=${OPTARG} ;;
    f) dlib_model=${OPTARG} ;;
    a) audio_model=${OPTARG} ;;
    ?) echo "Invalid option" -${OPTARG}; usage ;;
  esac
done

shift $(( OPTIND - 1 ))

######################################################################

# Find dlib model
if [[ -z ${dlib_model} ]]; then
  dlib_model=$(find ${HOME}/models -name "${dlib_file}" | head -n 1)
fi

if [[ -z ${dlib_model} ]]; then
  echo "Cannot find ${dlib_file}"
  echo "File should be in ~/models/dlib"
  echo "Or specify a file with -f option"
  exit 1
fi

if [[ ! -f ${dlib_model} ]]; then
  echo "File <${dlib}> not found"
  exit 1
fi

# Find audio model
if [[ -z ${audio_model} ]]; then
  audio_model=$(find ${HOME}/models -name "${audio_file}" | head -n 1)
fi

if [[ -z ${audio_model} ]]; then
  echo "Cannot find ${audio_file}"
  echo "File should be in ~/models/audio"
  echo "Or specify a file with -a option"
  exit 1
fi

if [[ ! -f ${audio_model} ]]; then
  echo "File <${audio_model}> not found"
  exit 1
fi

# Remove extension .model.pkl
audio_model=${audio_model%.model.pkl}

######################################################################

SESSION="perception"

echo "setup <${setup}>"
echo "config <${config}>"
echo "dlib model <${dlib_model}>"
echo "audio model <${audio_model}>"

if [[ "${setup}" = "pepper" ]]; then
  color="/naoqi_driver_node/camera/front/image_color"
  depth="/naoqi_driver_node/camera/depth_registered/image_rect"
  depth_config="/naoqi_driver_node/camera/depth_registered/camera_info"
elif [[ "${setup}" = "zr300" ]]; then
  color="/camera/rgb/image_rect_color"
  depth="/camera/depth_registered/sw_registered/image_rect_raw"
  depth_config="/camera/depth_registered/sw_registered/camera_info"
elif [[ "${setup}" = "d400" ]]; then
  # Old version developement branch of intel-ros/realsense
  # color="/camera/color/image_rect_color" #"/camera/color/image_raw"
  # depth="/camera/aligned_depth_to_color/image_raw" #"/camera/depth/image_rect_raw"
  # depth_config="/camera/depth/camera_info"
  # Master branch realsense2_camera
  color="/camera/color/image_raw"
  depth="/camera/aligned_depth_to_color/image_raw"
  depth_config="/camera/aligned_depth_to_color/camera_info"
elif [[ "${setup}" = "kinect" ]]; then
  color="/kinect2/qhd/image_color_rect"
  depth="/kinect2/qhd/image_depth_rect"
  depth_config="/kinect2/qhd/camera_info"
elif [[ "${setup}" = "webcam" ]]; then
  color="/cv_camera/image_raw"
else
  echo "Unknown setup"
  exit 1
fi

echo "session <${SESSION}>"

check_session=`tmux ls | grep ${SESSION}`
if [ -n "$check_session" ]; then
  echo -e "\e[1m\e[31m+++ ERROR: tmux session with name $SESSION is already running. +++\e[0m"
  echo "tmux a -t ${SESSION}"
  exit 1
fi

# Split window in 4
# - 0 : roscore
# - 1 : top
# - 2 : rviz
# - 3 :
tmux new-session -d -s ${SESSION} "bash"
sleep 2
tmux split-window -v
tmux select-pane -t 0
tmux split-window -h
tmux select-pane -t 2
tmux split-window -h

tmux select-pane -t 1
tmux send-keys "top -d 0.2" "C-m"
tmux select-pane -t 2
if [[ "${setup}" = "pepper" ]]; then
  tmux send-keys "rosrun rviz rviz -d $(rospack find person_manager)/rviz/pepper.rviz"
else
  tmux send-keys "rosrun rviz rviz"
fi

# Camera/Robot
tmux new-window -t ${SESSION} -n "camera"

if [[ "${setup}" = "pepper" ]]; then
  tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver naoqi_driver.launch"
  tmux split-window -v
  tmux select-pane -t 1
  tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver register_depth.launch"
  tmux select-pane -t 0
elif [[ "${setup}" = "zr300" ]]; then
  tmux send-keys "roslaunch realsense_camera zr300_nodelet_rgdb.launch"
elif [[ "${setup}" = "d400" ]]; then
  # tmux send-keys "roslaunch realsense_ros_camera rs_camera.launch"
  #
  # rs_camera.launch is very computationnaly expensive so we force it
  # to run on 1 core only with taskset

  # color_width:=$W color_height:=$H depth_width:=$W depth_height:=$H
  # infra1_width:=$W infra1_height:=$H infra2_width:=$W
  # infra2_height:=$H

  # tmux send-keys "taskset -c 0 roslaunch realsense_ros_camera rs_camera.launch"
  W=1280
  H=720
  tmux send-keys "roslaunch realsense2_camera rs_camera.launch align_depth:=1 enable_sync:=1 color_width:=$W color_height:=$H depth_width:=$W depth_height:=$H infra1_width:=$W infra1_height:=$H infra2_width:=$W infra2_height:=$H"

elif [[ "${setup}" = "kinect" ]]; then
  width=480
  height=260
  tmux send-keys "roslaunch kinect2_bridge kinect2_bridge.launch publish_tf:=1 calib_path:=${HOME}/calib worker_threads:=2 fps_limit:=20"
elif [[ "${setup}" = "webcam" ]]; then
  tmux send-keys "rosrun cv_camera cv_camera_node"
fi

# Perception nodes
tmux new-window -t ${SESSION} -n "perception"

tmux send-keys "GLOG_minloglevel=2 roslaunch particle_person_tracker main.launch color:=${color} config:=$(catkin_find particle_person_tracker ${config} | head -n 1) delay:=${delay} particles:=${particules} height:=${height} width:=${width} # opts:='--nogui' "

if [[ "${setup}" = "pepper" ]]; then
  tmux split-window -v
  # tmux select-pane -t 1
  tmux send-keys "rosrun audio_perception ssl_nn.py -n stft --context-frames=25 --frame-id=Head ~/models/audio/rec_stft_resnetctx_c25a0_1task"
fi


tmux split-window -v
# tmux select-pane -t 2
tmux send-keys "rosrun person_manager person_manager_node.py --color ${color} --depth ${depth} --info ${depth_config} --dlib ${dlib_model} --keep-threshold ${keep_threshold} --reid-threshold ${reid_threshold}"

tmux split-window -v
# tmux select-pane -t 3

tmux select-layout even-vertical

tmux send-keys "rosrun person_manager viewer_node.py --color ${color} --visu 1 --markers 1"

tmux select-pane -t 0


# Go back to first window
tmux select-window -t ${SESSION}:0
tmux attach-session -t ${SESSION}

# # Run roscore
# tmux -2 new-session -d -s ${SESSION}
# tmux rename-window -t ${SESSION} "roscore"
# tmux select-window -t ${SESSION}:roscore
# tmux send-keys "roscore" C-m
