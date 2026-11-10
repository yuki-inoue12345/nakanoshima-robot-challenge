# Read-only snapshot. This message never commands navigation.
uint8 SCHEMA_VERSION=1

uint8 ACTIVITY_STARTING=0
uint8 ACTIVITY_NAVIGATING=1
uint8 ACTIVITY_TRANSITIONING=2
uint8 ACTIVITY_TASK_WAITING=3
uint8 ACTIVITY_MANUAL_PAUSED=4
uint8 ACTIVITY_PERSON_SEARCH_STOPPING=5
uint8 ACTIVITY_PERSON_SEARCH_WAITING=6
uint8 ACTIVITY_PERSON_SEARCH_FAULT=7
uint8 ACTIVITY_HOLDING_STOP=8
uint8 ACTIVITY_FINISHED=9
uint8 ACTIVITY_SHUTTING_DOWN=10

# header.stamp is the snapshot creation time. frame_id is the route map frame.
std_msgs/Header header
uint8 schema_version

# SHA-256 of the exact waypoint YAML bytes loaded by the follower.
string route_sha256

# Zero-based index into that YAML list and its declared waypoint number.
uint32 next_waypoint_index
int64 next_waypoint_number

# "to_pose" or "through_poses"; copied from the active follower mode.
string navigation_mode

# Existing follower goal_serial. Changes invalidate outstanding reference plans.
uint64 goal_generation
uint8 activity
bool reference_allowed

# Inclusive zero-based task boundary computed by the follower's own logic.
uint32 safe_section_end_index
uint32 waypoint_count
