#include <memory>
#include <vector>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_utils.hpp"
#include "nav2_util/robot_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robotbase_bt_nodes/passage.hpp"
#include "tf2_ros/buffer.h"

namespace robotbase_bt_nodes
{
class RemoveTraversedGoals : public BT::SyncActionNode
{
  using Goals = std::vector<geometry_msgs::msg::PoseStamped>;

public:
  RemoveTraversedGoals(const std::string & name, const BT::NodeConfiguration & config)
  : BT::SyncActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<Goals>("input_goals"), BT::OutputPort<Goals>("output_goals"),
      BT::InputPort<std::string>("robot_base_frame", "Robot base frame")};
  }

private:
  BT::NodeStatus tick() override
  {
    Goals goals;
    if (!getInput("input_goals", goals)) {
      return BT::NodeStatus::FAILURE;
    }
    if (goals.size() < 2) {
      setOutput("output_goals", goals);
      return BT::NodeStatus::SUCCESS;
    }
    auto node = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
    auto tf = config().blackboard->get<std::shared_ptr<tf2_ros::Buffer>>("tf_buffer");
    const auto base = BT::deconflictPortAndParamFrame<std::string>(
      node, "robot_base_frame", this);
    double tolerance;
    node->get_parameter("transform_tolerance", tolerance);
    geometry_msgs::msg::PoseStamped pose;
    if (!nav2_util::getCurrentPose(pose, *tf, goals.front().header.frame_id, base, tolerance)) {
      return BT::NodeStatus::FAILURE;
    }
    const Point robot{pose.pose.position.x, pose.pose.position.y};
    // All poses of the follower's action share its dispatch stamp. A new action
    // (including resume/loop) starts a new anchor, even when the BT is reused.
    if (!initialized_ || stamp_ != goals.front().header.stamp || frame_ != goals.front().header.frame_id) {
      previous_ = robot;
      stamp_ = goals.front().header.stamp;
      frame_ = goals.front().header.frame_id;
      initialized_ = true;
    }
    while (goals.size() > 1) {
      const auto & g = goals[0].pose.position;
      const auto & n = goals[1].pose.position;
      if (!traversed(previous_, {g.x, g.y}, {n.x, n.y}, robot)) {
        break;
      }
      previous_ = {g.x, g.y};
      goals.erase(goals.begin());
    }
    setOutput("output_goals", goals);
    return BT::NodeStatus::SUCCESS;
  }

  bool initialized_{false};
  Point previous_{};
  builtin_interfaces::msg::Time stamp_;
  std::string frame_;
};
}  // namespace robotbase_bt_nodes

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<robotbase_bt_nodes::RemoveTraversedGoals>("RemoveTraversedGoals");
}
