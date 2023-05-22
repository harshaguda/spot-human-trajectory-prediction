// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from rclpy_message_converter_msgs:msg/TestArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "rclpy_message_converter_msgs/msg/detail/test_array__rosidl_typesupport_introspection_c.h"
#include "rclpy_message_converter_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "rclpy_message_converter_msgs/msg/detail/test_array__functions.h"
#include "rclpy_message_converter_msgs/msg/detail/test_array__struct.h"


// Include directives for member types
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  rclpy_message_converter_msgs__msg__TestArray__init(message_memory);
}

void rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_fini_function(void * message_memory)
{
  rclpy_message_converter_msgs__msg__TestArray__fini(message_memory);
}

size_t rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__size_function__TestArray__data(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_const_function__TestArray__data(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_function__TestArray__data(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__fetch_function__TestArray__data(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_const_function__TestArray__data(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__assign_function__TestArray__data(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_function__TestArray__data(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__resize_function__TestArray__data(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_member_array[1] = {
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(rclpy_message_converter_msgs__msg__TestArray, data),  // bytes offset in struct
    NULL,  // default value
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__size_function__TestArray__data,  // size() function pointer
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_const_function__TestArray__data,  // get_const(index) function pointer
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__get_function__TestArray__data,  // get(index) function pointer
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__fetch_function__TestArray__data,  // fetch(index, &value) function pointer
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__assign_function__TestArray__data,  // assign(index, value) function pointer
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__resize_function__TestArray__data  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_members = {
  "rclpy_message_converter_msgs__msg",  // message namespace
  "TestArray",  // message name
  1,  // number of fields
  sizeof(rclpy_message_converter_msgs__msg__TestArray),
  rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_member_array,  // message members
  rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_init_function,  // function to initialize message memory (memory has to be allocated)
  rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_type_support_handle = {
  0,
  &rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_rclpy_message_converter_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, rclpy_message_converter_msgs, msg, TestArray)() {
  if (!rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_type_support_handle.typesupport_identifier) {
    rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &rclpy_message_converter_msgs__msg__TestArray__rosidl_typesupport_introspection_c__TestArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
