// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from rclpy_message_converter_msgs:msg/TestArray.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "rclpy_message_converter_msgs/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "rclpy_message_converter_msgs/msg/detail/test_array__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace rclpy_message_converter_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_serialize(
  const rclpy_message_converter_msgs::msg::TestArray & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  rclpy_message_converter_msgs::msg::TestArray & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
get_serialized_size(
  const rclpy_message_converter_msgs::msg::TestArray & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
max_serialized_size_TestArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace rclpy_message_converter_msgs

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, msg, TestArray)();

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
