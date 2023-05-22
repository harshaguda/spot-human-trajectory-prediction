// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__BUILDER_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace rclpy_message_converter_msgs
{

namespace srv
{

namespace builder
{

class Init_NestedUint8ArrayTestService_Request_input
{
public:
  Init_NestedUint8ArrayTestService_Request_input()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request input(::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request::_input_type arg)
  {
    msg_.input = std::move(arg);
    return std::move(msg_);
  }

private:
  ::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request>()
{
  return rclpy_message_converter_msgs::srv::builder::Init_NestedUint8ArrayTestService_Request_input();
}

}  // namespace rclpy_message_converter_msgs


namespace rclpy_message_converter_msgs
{

namespace srv
{

namespace builder
{

class Init_NestedUint8ArrayTestService_Response_output
{
public:
  Init_NestedUint8ArrayTestService_Response_output()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response output(::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response::_output_type arg)
  {
    msg_.output = std::move(arg);
    return std::move(msg_);
  }

private:
  ::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response>()
{
  return rclpy_message_converter_msgs::srv::builder::Init_NestedUint8ArrayTestService_Response_output();
}

}  // namespace rclpy_message_converter_msgs

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__BUILDER_HPP_
