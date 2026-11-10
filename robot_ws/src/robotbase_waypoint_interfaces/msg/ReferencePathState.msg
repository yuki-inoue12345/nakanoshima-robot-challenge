# Atomic, read-only reference path and the follower state that authorized it.
# Consumers must reject the path unless this metadata still matches the current
# WaypointFollowerState. This message never commands navigation.
uint8 SCHEMA_VERSION=1

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
