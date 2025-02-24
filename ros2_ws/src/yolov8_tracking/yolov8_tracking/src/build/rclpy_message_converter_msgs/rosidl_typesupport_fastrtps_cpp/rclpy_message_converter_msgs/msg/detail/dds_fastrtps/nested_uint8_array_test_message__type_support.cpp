// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from rclpy_message_converter_msgs:msg/NestedUint8ArrayTestMessage.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__rosidl_typesupport_fastrtps_cpp.hpp"
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace rclpy_message_converter_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage &);
size_t get_serialized_size(
  const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage &,
  size_t current_alignment);
size_t
max_serialized_size_Uint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace rclpy_message_converter_msgs


namespace rclpy_message_converter_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_serialize(
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: arrays
  {
    size_t size = ros_message.arrays.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.arrays[i],
        cdr);
    }
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage & ros_message)
{
  // Member: arrays
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.arrays.resize(size);
    for (size_t i = 0; i < size; i++) {
      rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.arrays[i]);
    }
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
get_serialized_size(
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: arrays
  {
    size_t array_size = ros_message.arrays.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.arrays[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
max_serialized_size_NestedUint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: arrays
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      current_alignment +=
        rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Uint8ArrayTestMessage(
        inner_full_bounded, inner_is_plain, current_alignment);
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  return current_alignment - initial_alignment;
}

static bool _NestedUint8ArrayTestMessage__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _NestedUint8ArrayTestMessage__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _NestedUint8ArrayTestMessage__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _NestedUint8ArrayTestMessage__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_NestedUint8ArrayTestMessage(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _NestedUint8ArrayTestMessage__callbacks = {
  "rclpy_message_converter_msgs::msg",
  "NestedUint8ArrayTestMessage",
  _NestedUint8ArrayTestMessage__cdr_serialize,
  _NestedUint8ArrayTestMessage__cdr_deserialize,
  _NestedUint8ArrayTestMessage__get_serialized_size,
  _NestedUint8ArrayTestMessage__max_serialized_size
};

static rosidl_message_type_support_t _NestedUint8ArrayTestMessage__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_NestedUint8ArrayTestMessage__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace rclpy_message_converter_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_rclpy_message_converter_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage>()
{
  return &rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::_NestedUint8ArrayTestMessage__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage)() {
  return &rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::_NestedUint8ArrayTestMessage__handle;
}

#ifdef __cplusplus
}
#endif
