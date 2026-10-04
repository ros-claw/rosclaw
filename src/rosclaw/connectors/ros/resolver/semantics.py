"""Type-based ROS1/ROS2 semantics; names alone never establish implementation."""

SEMANTIC_TYPES = {
    "sensor_msgs/msg/LaserScan": "sensing.lidar",
    "sensor_msgs/LaserScan": "sensing.lidar",
    "sensor_msgs/msg/PointCloud2": "sensing.pointcloud",
    "nav_msgs/msg/Odometry": "state.odometry",
    "nav_msgs/Odometry": "state.odometry",
    "nav_msgs/msg/OccupancyGrid": "mapping.occupancy_map",
    "nav_msgs/OccupancyGrid": "mapping.occupancy_map",
    "geometry_msgs/msg/PoseWithCovarianceStamped": "localization.pose_estimate",
    "geometry_msgs/PoseWithCovarianceStamped": "localization.pose_estimate",
    "nav2_msgs/action/NavigateToPose": "navigation.navigate_to_pose",
    "move_base_msgs/MoveBaseAction": "navigation.navigate_to_pose",
    "nav2_msgs/action/NavigateThroughPoses": "navigation.navigate_through_poses",
    "nav2_msgs/action/ComputePathToPose": "navigation.compute_path",
    "opennav_coverage_msgs/action/ComputeCoveragePath": "coverage.compute_path",
    "opennav_coverage_msgs/action/NavigateCompleteCoverage": "coverage.execute",
}


def semantic_id(ros_type: str) -> str | None:
    return SEMANTIC_TYPES.get(ros_type)
