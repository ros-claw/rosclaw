"""Composite probes with frozen, independently reviewable offline witnesses."""

HARD_CASES = [
    (
        "camera_extrinsic_depth_units",
        "Depth mm -> camera xyz using pinhole -> world=R*camera+t. Return world_xyz_m for each pixel, null for invalid depth0.",
        {
            "pixels": [[420, 240, 2000], [320, 240, 0]],
            "fx": 500,
            "fy": 500,
            "cx": 320,
            "cy": 240,
            "R": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "t": [1, 2, 3],
        },
        {"world_xyz_m": [[1, 2.4, 5], None]},
    ),
    (
        "ros2_discovery_qos_namespace",
        "Resolve subscriber topic under namespace. Connect only matching resolved topic/domain and reliable publisher. Return matched_ids.",
        {
            "subscriber": {"topic": "scan", "namespace": "/g1", "domain": 42},
            "publishers": [
                {"id": "a", "topic": "/g1/scan", "domain": 42, "reliability": "best_effort"},
                {"id": "b", "topic": "/g1/scan", "domain": 42, "reliability": "reliable"},
                {"id": "c", "topic": "/g1/scan", "domain": 0, "reliability": "reliable"},
                {"id": "d", "topic": "/scan", "domain": 42, "reliability": "reliable"},
            ],
        },
        {"matched_ids": ["b"]},
    ),
    (
        "sensor_clock_offset_sync",
        "Camera stamp plus offset gives common time. For each choose nearest lidar within4ms, ties earlier, no reuse. Return lidar_match_ms/null.",
        {"camera_ms": [100, 110, 120], "offset_ms": 20, "lidar_ms": [118, 122, 135]},
        {"lidar_match_ms": [118, None, None]},
    ),
    (
        "rosbag_reset_epoch_sequence",
        "Start new epoch when timestamp decreases; count absent sequence ids only within epoch, duplicates not lost. Return epoch_start_indices,total_missing.",
        {
            "records": [
                {"t": 100, "seq": 1},
                {"t": 110, "seq": 4},
                {"t": 5, "seq": 1},
                {"t": 6, "seq": 1},
                {"t": 7, "seq": 3},
            ]
        },
        {"epoch_start_indices": [0, 2], "total_missing": 3},
    ),
    (
        "actuator_aggregate_joint_clip",
        "First clamp each request to actuator bounds, multiply by gear, sum into joint, clamp joint bound. Return actuator_forces,requested_joint_torque,applied_joint_torque.",
        {
            "requests": [5, -4],
            "actuator_bounds": [[-2, 3], [-5, 2]],
            "gears": [2, -3],
            "joint_bounds": [-10, 12],
        },
        {"actuator_forces": [3, -4], "requested_joint_torque": 18, "applied_joint_torque": 12},
    ),
    (
        "filter_predict_two_measurements",
        "Before each measurement P+=Q; K=P/(P+R); x+=K*(z-x); P=(1-K)*P. Return final_x,final_P.",
        {"x": 0, "P": 1, "Q": 1, "R": 2, "measurements": [2, 4]},
        {"final_x": 2.5, "final_P": 1},
    ),
    (
        "pointcloud_big_endian_padded",
        "Big-endian float32 x,y offsets0,4, point_step12 with padding. Reject y<0. Return accepted_xy.",
        {"point_step": 12, "hex": "3f80000040000000deadbeef40400000bf800000cafebabe"},
        {"accepted_xy": [[1, 2]]},
    ),
    (
        "vla_chunk_order_units_stale",
        "Accept chunks in arrival order only if age<=100ms and generation >= last accepted generation. Reorder policy to body names and degrees to radians. Return accepted_ids,last_action_rad.",
        {
            "policy_names": ["knee", "hip"],
            "body_names": ["hip", "knee"],
            "now_ms": 1000,
            "chunks": [
                {"id": "a", "generated_ms": 850, "action_deg": [90, 0]},
                {"id": "b", "generated_ms": 960, "action_deg": [180, 90]},
                {"id": "c", "generated_ms": 950, "action_deg": [0, 0]},
            ],
        },
        {"accepted_ids": ["b"], "last_action_rad": [1.5707963267948966, 3.141592653589793]},
    ),
    (
        "navigation_latency_moving_obstacle",
        "Required clearance = v^2/(2a)+v*latency + obstacle_speed*(v/a+latency). Safe iff clearance>=required. Return required_clearance_m,safe.",
        {"v": 2, "a": 1, "latency": 0.5, "obstacle_speed": 0.4, "clearance": 3.9},
        {"required_clearance_m": 4, "safe": False},
    ),
    (
        "wrench_sensor_to_joint",
        "World F=R*F_sensor, sensor tau_world=R*tau_sensor; joint torque=sensor tau_world+r_joint_to_sensor cross world F. Return world_F,joint_tau.",
        {
            "R": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "F_sensor": [2, 0, 0],
            "tau_sensor": [0, 0, 1],
            "r_joint_to_sensor": [1, 0, 0],
        },
        {"world_F": [0, 2, 0], "joint_tau": [0, 0, 3]},
    ),
    (
        "vln_landmark_temporal_staleness",
        "Only observations age<=100ms in current room eligible. Need unique red door otherwise clarify. Return eligible_ids,decision.",
        {
            "now_ms": 1000,
            "room": "A",
            "objects": [
                {"id": "old", "room": "A", "stamp_ms": 800, "color": "red", "type": "door"},
                {"id": "wrong_room", "room": "B", "stamp_ms": 990, "color": "red", "type": "door"},
                {"id": "live", "room": "A", "stamp_ms": 950, "color": "red", "type": "door"},
            ],
        },
        {"eligible_ids": ["live"], "decision": "execute"},
    ),
    (
        "trajectory_named_channels_effort",
        "Reorder efforts by body joint names; reject whole trajectory if any absolute effort exceeds that joint bound or time not strictly increasing. Return valid,first_failure_point.",
        {
            "body_names": ["hip", "knee"],
            "bounds": [2, 5],
            "trajectory_names": ["knee", "hip"],
            "points": [
                {"t": 0, "effort": [4, 1]},
                {"t": 1, "effort": [1, 3]},
                {"t": 2, "effort": [0, 0]},
            ],
        },
        {"valid": False, "first_failure_point": 1},
    ),
]
