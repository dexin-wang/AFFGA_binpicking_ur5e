cmake_minimum_required(VERSION 2.8.12)
project(AFFGA_binpicking_ur5e)

add_definitions( -DROS_BUILD )

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message("${PROJECT_NAME}: You did not request a specific build type: selecting 'RelWithDebInfo'.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(catkin REQUIRED
  COMPONENTS
    actionlib
    control_msgs
    controller_manager
    controller_manager_msgs
    geometry_msgs
    hardware_interface
    industrial_robot_status_interface
    pluginlib
    roscpp
    scaled_joint_trajectory_controller
    sensor_msgs
    speed_scaling_interface
    speed_scaling_state_controller
    std_srvs
    tf
    tf2_geometry_msgs
    tf2_msgs
    trajectory_msgs
    ur_dashboard_msgs
    ur_msgs
    pass_through_controllers
    kdl_parser
    message_generation
)
find_package(Boost REQUIRED)
find_package(ur_client_library REQUIRED)


add_message_files(
  FILES
  graspPose.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  # INCLUDE_DIRS
    # include
  # LIBRARIES
  #   ur_robot_driver_plugin
  #   urcl_log_handler
  CATKIN_DEPENDS
    actionlib
    control_msgs
    controller_manager
    controller_manager_msgs
    geometry_msgs
    hardware_interface
    kdl_parser
    pass_through_controllers
    pluginlib
    roscpp
    scaled_joint_trajectory_controller
    sensor_msgs
    speed_scaling_interface
    speed_scaling_state_controller
    std_srvs
    tf
    tf2_geometry_msgs
    tf2_msgs
    trajectory_msgs
    ur_dashboard_msgs
    ur_msgs
    message_runtime
  DEPENDS
    Boost
    ur_client_library
)

add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-Wno-unused-parameter)



catkin_install_python(PROGRAMS scripts/test_move.py scripts/policy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
# )

# install(FILES hardware_interface_plugin.xml
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
