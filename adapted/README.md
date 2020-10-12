# Hack to automatically start Intel camera and reboot Jetson when crashed

1. Copy `autostart_intel.sh` and `monitor_intel_node.py` to the home of the jetson:

    ```
    scp autostart_intel.sh monitor_intel_node.py nvidia@${JETSON_IP}:~
    ```

2. Make the `nvidia` user being able to reboot without password: on
the Jetson, run `sudo visudo` and add at the end of the file the
following line:

    ```
    nvidia ALL=NOPASSWD:/sbin/reboot
    ```

3. Create a cronjob that will run `autostart_intel.sh` on startup:

    ```bash
    crontab -e -u nvidia
    ```

    and add

    ```
    SHELL=/bin/bash
    @reboot bash ~nvidia/autostart_intel.sh
    ```

4. When the Jetson is started, and when it detects that the roscore is
running on the remote laptop, the 2 scripts start. You can access them
with:

    ```bash
    tmux a -t camera
    ```

5. :warning: `All the environment variables` should be in `.bashrc`:

    ```
    export ROS_IP=
    export ROS_MASTER_URI=http://...:11311
    source ~/catkin_ws/devel/setup.bash
    ```
