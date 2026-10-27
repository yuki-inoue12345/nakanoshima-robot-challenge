#ifndef ROBOTBASE_BT_NODES__PASSAGE_HPP_
#define ROBOTBASE_BT_NODES__PASSAGE_HPP_

#include <algorithm>
#include <cmath>

namespace robotbase_bt_nodes
{
struct Point {double x; double y;};
inline Point subtract(Point a, Point b) {return {a.x - b.x, a.y - b.y};}
inline double dot(Point a, Point b) {return a.x * b.x + a.y * b.y;}
inline double segment_distance_squared(Point p, Point a, Point b)
{
  const auto v = subtract(b, a);
  const double length = dot(v, v);
  const double t = length > 1e-12 ? std::clamp(dot(subtract(p, a), v) / length, 0., 1.) : 0.;
  const auto error = subtract(p, {a.x + t * v.x, a.y + t * v.y});
  return dot(error, error);
}

// Only the next waypoint and its adjoining segments participate. There is no
// arrival radius: being one metre (or any distance) before a point never removes it.
inline bool traversed(Point previous, Point goal, Point next, Point robot)
{
  auto incoming = subtract(goal, previous);
  const auto outgoing = subtract(next, goal);
  if (dot(incoming, incoming) < 1e-12) {
    incoming = outgoing;
  }
  if (dot(incoming, incoming) < 1e-12) {
    return true;  // Coincident consecutive points describe no travel.
  }
  if (dot(subtract(robot, goal), incoming) >= 0.) {
    return true;  // Crossed the plane through the goal along the incoming leg.
  }
  // A smooth corner may join the outgoing leg without crossing the incoming
  // plane. Only count it after its outgoing projection is forward and closer
  // than the incoming segment, rather than merely being near the corner.
  return dot(subtract(robot, goal), outgoing) > 0. &&
         segment_distance_squared(robot, goal, next) + 1e-12 <
         segment_distance_squared(robot, previous, goal);
}
}  // namespace robotbase_bt_nodes
#endif
