// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from rclpy_message_converter_msgs:msg/NestedUint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__BUILDER_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace rclpy_message_converter_msgs
{

namespace msg
{

namespace builder
{

class Init_NestedUint8ArrayTestMessage_arrays
{
public:
  Init_NestedUint8ArrayTestMessage_arrays()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage arrays(::rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage::_arrays_type arg)
  {
    msg_.arrays = std::move(arg);
    return std::move(msg_);
  }

private:
  ::rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage>()
{
  return rclpy_message_converter_msgs::msg::builder::Init_NestedUint8ArrayTestMessage_arrays();
}

}  // namespace rclpy_message_converter_msgs

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__BUILDER_HPP_
