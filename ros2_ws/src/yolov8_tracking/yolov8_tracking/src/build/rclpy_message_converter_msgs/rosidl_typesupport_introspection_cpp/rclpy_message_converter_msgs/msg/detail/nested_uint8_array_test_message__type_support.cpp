// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from rclpy_message_converter_msgs:msg/NestedUint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace rclpy_message_converter_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void NestedUint8ArrayTestMessage_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage(_init);
}

void NestedUint8ArrayTestMessage_fini_function(void * message_memory)
{
  auto typed_message = static_cast<rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage *>(message_memory);
  typed_message->~NestedUint8ArrayTestMessage();
}

size_t size_function__NestedUint8ArrayTestMessage__arrays(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage> *>(untyped_member);
  return member->size();
}

const void * get_const_function__NestedUint8ArrayTestMessage__arrays(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage> *>(untyped_member);
  return &member[index];
}

void * get_function__NestedUint8ArrayTestMessage__arrays(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage> *>(untyped_member);
  return &member[index];
}

void fetch_function__NestedUint8ArrayTestMessage__arrays(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage *>(
    get_const_function__NestedUint8ArrayTestMessage__arrays(untyped_member, index));
  auto & value = *reinterpret_cast<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage *>(untyped_value);
  value = item;
}

void assign_function__NestedUint8ArrayTestMessage__arrays(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage *>(
    get_function__NestedUint8ArrayTestMessage__arrays(untyped_member, index));
  const auto & value = *reinterpret_cast<const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage *>(untyped_value);
  item = value;
}

void resize_function__NestedUint8ArrayTestMessage__arrays(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember NestedUint8ArrayTestMessage_message_member_array[1] = {
  {
    "arrays",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage, arrays),  // bytes offset in struct
    nullptr,  // default value
    size_function__NestedUint8ArrayTestMessage__arrays,  // size() function pointer
    get_const_function__NestedUint8ArrayTestMessage__arrays,  // get_const(index) function pointer
    get_function__NestedUint8ArrayTestMessage__arrays,  // get(index) function pointer
    fetch_function__NestedUint8ArrayTestMessage__arrays,  // fetch(index, &value) function pointer
    assign_function__NestedUint8ArrayTestMessage__arrays,  // assign(index, value) function pointer
    resize_function__NestedUint8ArrayTestMessage__arrays  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers NestedUint8ArrayTestMessage_message_members = {
  "rclpy_message_converter_msgs::msg",  // message namespace
  "NestedUint8ArrayTestMessage",  // message name
  1,  // number of fields
  sizeof(rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage),
  NestedUint8ArrayTestMessage_message_member_array,  // message members
  NestedUint8ArrayTestMessage_init_function,  // function to initialize message memory (memory has to be allocated)
  NestedUint8ArrayTestMessage_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t NestedUint8ArrayTestMessage_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &NestedUint8ArrayTestMessage_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace rclpy_message_converter_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage>()
{
  return &::rclpy_message_converter_msgs::msg::rosidl_typesupport_introspection_cpp::NestedUint8ArrayTestMessage_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, rclpy_message_converter_msgs, msg, NestedUint8ArrayTestMessage)() {
  return &::rclpy_message_converter_msgs::msg::rosidl_typesupport_introspection_cpp::NestedUint8ArrayTestMessage_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
