// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rclpy_message_converter_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__struct.h"
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__functions.h"
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

#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__functions.h"  // input

// forward declare type support functions
size_t get_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage)();


using _NestedUint8ArrayTestService_Request__ros_msg_type = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request;

static bool _NestedUint8ArrayTestService_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _NestedUint8ArrayTestService_Request__ros_msg_type * ros_message = static_cast<const _NestedUint8ArrayTestService_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: input
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->input, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _NestedUint8ArrayTestService_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _NestedUint8ArrayTestService_Request__ros_msg_type * ros_message = static_cast<_NestedUint8ArrayTestService_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: input
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->input))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t get_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _NestedUint8ArrayTestService_Request__ros_msg_type * ros_message = static_cast<const _NestedUint8ArrayTestService_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name input

  current_alignment += get_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
    &(ros_message->input), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _NestedUint8ArrayTestService_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t max_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request(
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

  // member: input
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      current_alignment +=
        max_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
        inner_full_bounded, inner_is_plain, current_alignment);
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _NestedUint8ArrayTestService_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_NestedUint8ArrayTestService_Request = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService_Request",
  _NestedUint8ArrayTestService_Request__cdr_serialize,
  _NestedUint8ArrayTestService_Request__cdr_deserialize,
  _NestedUint8ArrayTestService_Request__get_serialized_size,
  _NestedUint8ArrayTestService_Request__max_serialized_size
};

static rosidl_message_type_support_t _NestedUint8ArrayTestService_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_NestedUint8ArrayTestService_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Request)() {
  return &_NestedUint8ArrayTestService_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "rclpy_message_converter_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__struct.h"
// already included above
// #include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

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

// already included above
// #include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__functions.h"  // output

// forward declare type support functions
size_t get_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage)();


using _NestedUint8ArrayTestService_Response__ros_msg_type = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response;

static bool _NestedUint8ArrayTestService_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _NestedUint8ArrayTestService_Response__ros_msg_type * ros_message = static_cast<const _NestedUint8ArrayTestService_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: output
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->output, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _NestedUint8ArrayTestService_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _NestedUint8ArrayTestService_Response__ros_msg_type * ros_message = static_cast<_NestedUint8ArrayTestService_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: output
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->output))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t get_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _NestedUint8ArrayTestService_Response__ros_msg_type * ros_message = static_cast<const _NestedUint8ArrayTestService_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name output

  current_alignment += get_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
    &(ros_message->output), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _NestedUint8ArrayTestService_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_rclpy_message_converter_msgs
size_t max_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response(
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

  // member: output
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      current_alignment +=
        max_serialized_size_rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage(
        inner_full_bounded, inner_is_plain, current_alignment);
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _NestedUint8ArrayTestService_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_NestedUint8ArrayTestService_Response = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService_Response",
  _NestedUint8ArrayTestService_Response__cdr_serialize,
  _NestedUint8ArrayTestService_Response__cdr_deserialize,
  _NestedUint8ArrayTestService_Response__get_serialized_size,
  _NestedUint8ArrayTestService_Response__max_serialized_size
};

static rosidl_message_type_support_t _NestedUint8ArrayTestService_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_NestedUint8ArrayTestService_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Response)() {
  return &_NestedUint8ArrayTestService_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rclpy_message_converter_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "rclpy_message_converter_msgs/srv/nested_uint8_array_test_service.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t NestedUint8ArrayTestService__callbacks = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Response)(),
};

static rosidl_service_type_support_t NestedUint8ArrayTestService__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &NestedUint8ArrayTestService__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService)() {
  return &NestedUint8ArrayTestService__handle;
}

#if defined(__cplusplus)
}
#endif
