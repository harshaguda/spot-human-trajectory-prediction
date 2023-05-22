// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from rclpy_message_converter_msgs:msg/Uint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__TRAITS_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "rclpy_message_converter_msgs/msg/detail/uint8_array_test_message__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace rclpy_message_converter_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Uint8ArrayTestMessage & msg,
  std::ostream & out)
{
  out << "{";
  // member: data
  {
    if (msg.data.size() == 0) {
      out << "data: []";
    } else {
      out << "data: [";
      size_t pending_items = msg.data.size();
      for (auto item : msg.data) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Uint8ArrayTestMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.data.size() == 0) {
      out << "data: []\n";
    } else {
      out << "data:\n";
      for (auto item : msg.data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Uint8ArrayTestMessage & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace rclpy_message_converter_msgs

namespace rosidl_generator_traits
{

[[deprecated("use rclpy_message_converter_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  rclpy_message_converter_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use rclpy_message_converter_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage & msg)
{
  return rclpy_message_converter_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>()
{
  return "rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage";
}

template<>
inline const char * name<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>()
{
  return "rclpy_message_converter_msgs/msg/Uint8ArrayTestMessage";
}

template<>
struct has_fixed_size<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__TRAITS_HPP_
