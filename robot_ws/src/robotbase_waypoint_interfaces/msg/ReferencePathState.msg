# Atomic, read-only reference path and the follower state that authorized it.
# Consumers must reject the path unless this metadata still matches the current
# WaypointFollowerState. This message never commands navigation.
uint8 SCHEMA_VERSION=2

std_msgs/Header header
uint8 schema_version
bool valid
string invalid_reason

string route_sha256
uint32 next_waypoint_index
int64 next_waypoint_number
string navigation_mode
uint64 goal_generation
uint32 safe_section_end_index
uint32 waypoint_count
uint64 planner_generation
builtin_interfaces/Time follower_state_stamp

float64 path_length_m
nav_msgs/Path path

# Final goal from the exact hash-validated waypoint YAML loaded by the
# reference Planner. This is map geometry only, not a planned route length.
bool final_waypoint_valid
int64 final_waypoint_number
geometry_msgs/Point final_waypoint_position
