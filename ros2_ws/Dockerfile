# =============================================================================
# ROS 2 Humble - Digital Twin Robotics Lab
# =============================================================================
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-foxglove-bridge \
    ros-humble-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install redis websockets

# Setup entrypoint
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

WORKDIR /ros2_ws
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
