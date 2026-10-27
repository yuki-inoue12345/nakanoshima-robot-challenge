cmake_minimum_required(VERSION 3.8)
project(robotbase_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY behavior_trees launch rviz urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
