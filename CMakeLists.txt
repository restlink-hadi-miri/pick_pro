cmake_minimum_required(VERSION 3.8)
project(o_detector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python package module (o_detector/)
ament_python_install_package(${PROJECT_NAME})

# Install the .py scripts so `ros2 run o_detector <name>.py` works
install(PROGRAMS
  scripts/ball_detector.py
  scripts/main_picker.py
  scripts/live_tcp_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

# Also install extension-less aliases so you can drop the ".py"
install(PROGRAMS scripts/ball_detector.py   DESTINATION lib/${PROJECT_NAME} RENAME ball_detector)
install(PROGRAMS scripts/main_picker.py     DESTINATION lib/${PROJECT_NAME} RENAME main_picker)
install(PROGRAMS scripts/live_tcp_logger.py DESTINATION lib/${PROJECT_NAME} RENAME live_tcp_logger)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

