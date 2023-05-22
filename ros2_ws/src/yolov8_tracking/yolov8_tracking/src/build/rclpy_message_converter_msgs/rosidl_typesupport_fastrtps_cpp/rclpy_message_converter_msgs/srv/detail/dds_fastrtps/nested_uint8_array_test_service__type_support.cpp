// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__rosidl_typesupport_fastrtps_cpp.hpp"
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__struct.hpp"

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
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &);
size_t get_serialized_size(
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &,
  size_t current_alignment);
size_t
max_serialized_size_NestedUint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace rclpy_message_converter_msgs


namespace rclpy_message_converter_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_serialize(
  const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: input
  rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.input,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request & ros_message)
{
  // Member: input
  rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.input);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
get_serialized_size(
  const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: input

  current_alignment +=
    rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.input, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
max_serialized_size_NestedUint8ArrayTestService_Request(
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


  // Member: input
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      current_alignment +=
        rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_NestedUint8ArrayTestMessage(
        inner_full_bounded, inner_is_plain, current_alignment);
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  return current_alignment - initial_alignment;
}

static bool _NestedUint8ArrayTestService_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _NestedUint8ArrayTestService_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _NestedUint8ArrayTestService_Request__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _NestedUint8ArrayTestService_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_NestedUint8ArrayTestService_Request(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _NestedUint8ArrayTestService_Request__callbacks = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService_Request",
  _NestedUint8ArrayTestService_Request__cdr_serialize,
  _NestedUint8ArrayTestService_Request__cdr_deserialize,
  _NestedUint8ArrayTestService_Request__get_serialized_size,
  _NestedUint8ArrayTestService_Request__max_serialized_size
};

static rosidl_message_type_support_t _NestedUint8ArrayTestService_Request__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_NestedUint8ArrayTestService_Request__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace rclpy_message_converter_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_rclpy_message_converter_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request>()
{
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService_Request__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Request)() {
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService_Request__handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include <limits>
// already included above
// #include <stdexcept>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
// already included above
// #include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace rclpy_message_converter_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &);
size_t get_serialized_size(
  const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage &,
  size_t current_alignment);
size_t
max_serialized_size_NestedUint8ArrayTestMessage(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace rclpy_message_converter_msgs


namespace rclpy_message_converter_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_serialize(
  const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: output
  rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.output,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response & ros_message)
{
  // Member: output
  rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.output);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
get_serialized_size(
  const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: output

  current_alignment +=
    rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.output, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_rclpy_message_converter_msgs
max_serialized_size_NestedUint8ArrayTestService_Response(
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


  // Member: output
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      current_alignment +=
        rclpy_message_converter_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_NestedUint8ArrayTestMessage(
        inner_full_bounded, inner_is_plain, current_alignment);
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  return current_alignment - initial_alignment;
}

static bool _NestedUint8ArrayTestService_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _NestedUint8ArrayTestService_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _NestedUint8ArrayTestService_Response__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _NestedUint8ArrayTestService_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_NestedUint8ArrayTestService_Response(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _NestedUint8ArrayTestService_Response__callbacks = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService_Response",
  _NestedUint8ArrayTestService_Response__cdr_serialize,
  _NestedUint8ArrayTestService_Response__cdr_deserialize,
  _NestedUint8ArrayTestService_Response__get_serialized_size,
  _NestedUint8ArrayTestService_Response__max_serialized_size
};

static rosidl_message_type_support_t _NestedUint8ArrayTestService_Response__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_NestedUint8ArrayTestService_Response__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace rclpy_message_converter_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_rclpy_message_converter_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response>()
{
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService_Response__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Response)() {
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService_Response__handle;
}

#ifdef __cplusplus
}
#endif

#include "rmw/error_handling.h"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/service_type_support_decl.hpp"

namespace rclpy_message_converter_msgs
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

static service_type_support_callbacks_t _NestedUint8ArrayTestService__callbacks = {
  "rclpy_message_converter_msgs::srv",
  "NestedUint8ArrayTestService",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService_Response)(),
};

static rosidl_service_type_support_t _NestedUint8ArrayTestService__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_NestedUint8ArrayTestService__callbacks,
  get_service_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace rclpy_message_converter_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_rclpy_message_converter_msgs
const rosidl_service_type_support_t *
get_service_type_support_handle<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService>()
{
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, rclpy_message_converter_msgs, srv, NestedUint8ArrayTestService)() {
  return &rclpy_message_converter_msgs::srv::typesupport_fastrtps_cpp::_NestedUint8ArrayTestService__handle;
}

#ifdef __cplusplus
}
#endif
