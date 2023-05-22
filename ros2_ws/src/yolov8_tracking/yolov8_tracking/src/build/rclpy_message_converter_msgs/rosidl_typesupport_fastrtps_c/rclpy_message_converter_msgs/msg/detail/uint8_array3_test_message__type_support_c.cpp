// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from rclpy_message_converter_msgs:msg/Uint8Array3TestMessage.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/msg/detail/uint8_array3_test_message__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rclpy_message_converter_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "rclpy_message_converter_msgs/msg/detail/uint8_array3_test_message__struct.h"
#include "rclpy_message_converter_msgs/msg/detail/uint8_array3_test_message__functions.h"
#include "fastcdr/Cdr.h"

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

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _Uint8Array3TestMessage__ros_msg_type = rclpy_message_converter_msgs__msg__Uint8Array3TestMessage;

static bool _Uint8Array3TestMessage__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Uint8Array3TestMessage__ros_msg_type * ros_message = static_cast<const _Uint8Array3TestMessage__ros_msg_type *>(untyped_ros_message);
  // Field name: data
  {
    size_t size = 3;
    auto array_ptr = ros_message->data;
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _Uint8Array3TestMessage__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Uint8Array3TestMessage__ros_msg_type * ros_message = static_cast<_Uint8Array3TestMessage__ros_msg_type *>(untyped_ros_message);
  // Field name: data
  {
    size_t size = 3;
    auto array_ptr = ros_message->data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t get_serialized_size_rclpy_message_converter_msgs__msg__Uint8Array3TestMessage(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Uint8Array3TestMessage__ros_msg_type * ros_message = static_cast<const _Uint8Array3TestMessage__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name data
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->data;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Uint8Array3TestMessage__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_rclpy_message_converter_msgs__msg__Uint8Array3TestMessage(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t max_serialized_size_rclpy_message_converter_msgs__msg__Uint8Array3TestMessage(
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

  // member: data
  {
    size_t array_size = 3;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _Uint8Array3TestMessage__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_rclpy_message_converter_msgs__msg__Uint8Array3TestMessage(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Uint8Array3TestMessage = {
  "rclpy_message_converter_msgs::msg",
  "Uint8Array3TestMessage",
  _Uint8Array3TestMessage__cdr_serialize,
  _Uint8Array3TestMessage__cdr_deserialize,
  _Uint8Array3TestMessage__get_serialized_size,
  _Uint8Array3TestMessage__max_serialized_size
};

static rosidl_message_type_support_t _Uint8Array3TestMessage__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Uint8Array3TestMessage,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, Uint8Array3TestMessage)() {
  return &_Uint8Array3TestMessage__type_support;
}

#if defined(__cplusplus)
}
#endif
