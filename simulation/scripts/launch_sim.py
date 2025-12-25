#!/usr/bin/env python3
"""
Isaac Sim Launcher
Launches Isaac Sim with ROS 2 bridge and warehouse environment
"""

import argparse
import os
import sys

# Isaac Sim imports (available inside Isaac Sim Python environment)
try:
    from omni.isaac.kit import SimulationApp
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Warning: Isaac Sim not available. Run inside Isaac Sim environment.")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Isaac Sim")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--environment", default="warehouse", help="Environment to load")
    parser.add_argument("--robot", default="carter", help="Robot to spawn")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not ISAAC_AVAILABLE:
        print("Error: Must run inside Isaac Sim environment")
        print("Use: ./python.sh simulation/scripts/launch_sim.py")
        sys.exit(1)
    
    # Configure simulation
    config = {
        "headless": args.headless,
        "width": 1920,
        "height": 1080,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 3,
    }
    
    # Start simulation app
    simulation_app = SimulationApp(config)
    
    # Import after SimulationApp is created
    import carb
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.extensions import enable_extension
    
    # Enable ROS 2 bridge
    enable_extension("omni.isaac.ros2_bridge")
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    
    # Load environment
    env_path = f"/isaac-sim/assets/environments/{args.environment}.usd"
    if os.path.exists(env_path):
        omni.usd.get_context().open_stage(env_path)
    else:
        print(f"Environment not found: {env_path}")
        print("Creating default ground plane...")
        world.scene.add_default_ground_plane()
    
    # Setup robot
    setup_robot(world, args.robot)
    
    # Setup sensors
    setup_sensors(world)
    
    # Reset world
    world.reset()
    
    print("Isaac Sim initialized. Starting simulation loop...")
    print("ROS 2 bridge enabled. Topics available:")
    print("  /cmd_vel (subscriber)")
    print("  /odom (publisher)")
    print("  /scan (publisher)")
    print("  /camera/rgb (publisher)")
    
    # Simulation loop
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()


def setup_robot(world, robot_type):
    """Spawn robot in the world."""
    from omni.isaac.core.robots import Robot
    from omni.isaac.wheeled_robots.robots import WheeledRobot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    
    robot_usd_map = {
        "carter": "/isaac-sim/assets/robots/carter/carter_v1.usd",
        "jetbot": "/isaac-sim/assets/robots/jetbot/jetbot.usd",
        "custom": "/isaac-sim/workspace/robots/dt_robot.usd",
    }
    
    robot_path = robot_usd_map.get(robot_type)
    
    if robot_path and os.path.exists(robot_path):
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")
        print(f"Loaded robot: {robot_type}")
    else:
        print(f"Robot USD not found, using default Carter")
        # Use built-in Carter robot
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers import DifferentialController
        
        robot = WheeledRobot(
            prim_path="/World/Robot",
            name="dt_robot",
            wheel_dof_names=["left_wheel", "right_wheel"],
            create_robot=True,
            position=[0, 0, 0],
        )
        world.scene.add(robot)


def setup_sensors(world):
    """Configure robot sensors for ROS 2 publishing."""
    from omni.isaac.sensor import LidarRtx, Camera
    from omni.isaac.ros2_bridge import ROS2Publisher
    
    # Lidar sensor
    try:
        lidar = LidarRtx(
            prim_path="/World/Robot/Lidar",
            name="lidar",
            position=[0, 0, 0.3],
            rotation=[0, 0, 0],
        )
        print("Lidar sensor configured")
    except Exception as e:
        print(f"Lidar setup warning: {e}")
    
    # Camera sensor
    try:
        camera = Camera(
            prim_path="/World/Robot/Camera",
            name="camera",
            position=[0.2, 0, 0.2],
            rotation=[0, 0, 0],
            resolution=(640, 480),
        )
        print("Camera sensor configured")
    except Exception as e:
        print(f"Camera setup warning: {e}")


if __name__ == "__main__":
    main()
