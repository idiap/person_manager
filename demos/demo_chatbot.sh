#!/bin/bash

SESSION=idiap

LANGUAGE="English"
MODE="base"
echo $LANGUAGE
echo $MODE

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
tmux rename-window -t $SESSION:0 'core|mongo'
tmux new-window -t $SESSION:1 -n 'robot'
tmux new-window -t $SESSION:2 -n 'asr|tts'
tmux new-window -t $SESSION:3 -n 'planners'
tmux new-window -t $SESSION:4 -n 'actions'
tmux new-window -t $SESSION:5 -n 'ontology'
tmux new-window -t $SESSION:6 -n 'alana'
tmux new-window -t $SESSION:7 -n 'perception'
tmux new-window -t $SESSION:8 -n 'ssp'
tmux new-window -t $SESSION:9 -n 'controller'
tmux new-window -t $SESSION:10 -n 'watchdog'

tmux select-window -t $SESSION:0
tmux split-window -v
tmux select-pane -t 0
tmux split-window -h
tmux select-pane -t 0
tmux send-keys "roscore" C-m
tmux select-pane -t 1
tmux send-keys "DISPLAY=:0 roslaunch mongodb_store mongodb_store.launch"
tmux select-pane -t 2
tmux send-keys "htop" C-m
tmux select-pane -t 1

tmux select-window -t $SESSION:1
tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver naoqi_driver.launch network_interface:=enp5s0"
if [ $MODE == "base" ]; then
    tmux split-window -h
    tmux select-pane -t 1
    tmux send-keys "rosservice call /naoqi_driver/motion/wake_up"
    tmux split-window -h
    tmux select-pane -t 2
    tmux send-keys "DISPLAY=:0 roslaunch naoqi_driver register_depth.launch"
fi
tmux select-pane -t 0

tmux select-window -t $SESSION:2
tmux split-window -h
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch mummer_asr_launch mummer_asr.launch use_extra_mic:=false target_language:=en-UK"
tmux select-pane -t 1
tmux send-keys "DISPLAY=:0 roslaunch dialogue_say dialogue_say.launch"
tmux select-pane -t 0

tmux select-window -t $SESSION:3
tmux split-window -h
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch dialogue_arbiter dialogue_arbiter.launch test:=false target_language:=English"
tmux select-pane -t 1
tmux send-keys "DISPLAY=:0 roslaunch rpn_recipe_planner recipe_planner.launch world:=mummer"
tmux select-pane -t 0

tmux select-window -t $SESSION:4
tmux split-window -h
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch rpn_recipe_planner example_servers.launch"
tmux select-pane -t 1
tmux send-keys 'DISPLAY=:0 roslaunch dialogue_arbiter example_servers.launch'
tmux select-pane -t 0

tmux select-window -t $SESSION:5
tmux split-window -h
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch semantic_route_description ideapark.launch"
tmux select-pane -t 1
tmux send-keys "DISPLAY=:0 rosrun semantic_route_description description"
tmux select-pane -t 0

tmux select-window -t $SESSION:6
tmux send-keys "DISPLAY=:0 roslaunch chatbot_interface alana_interface.launch project_id:=ALANA_DIRECT"

tmux select-window -t $SESSION:7
tmux split-window -v
tmux select-pane -t 0
tmux send-keys "DISPLAY=:0 roslaunch person_manager perception.launch naoqi:=1 tracker_delay:=50 tracker_scale:=0.5 tracker_detector:=openheadpose tracker_particles:=50 tracker_min_height:=0.12 manager_keep_threshold:=0.02 tracker_visu:=0 manager_visu:=1 with_audio:=1"
tmux select-pane -t 1
tmux send-keys "rosrun person_manager demo_stimulus_reaction.py -m 0.15"

tmux select-window -t $SESSION:8
tmux send-keys "DISPLAY=:0 roslaunch attention_analyser attention_analyser_mm.launch"

tmux select-window -t $SESSION:9
tmux send-keys "DISPLAY=:0 roslaunch pepper_planning_control recipe_controller.launch laas:=false"

# Set default window
tmux select-window -t $SESSION:0
#tmux setw -g mode-mouse off

# Attach to session
tmux -2 attach-session -t $SESSION
