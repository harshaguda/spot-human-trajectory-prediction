// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from rclpy_message_converter_msgs:msg/TestArray.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'data'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/TestArray in the package rclpy_message_converter_msgs.
typedef struct rclpy_message_converter_msgs__msg__TestArray
{
  rosidl_runtime_c__double__Sequence data;
} rclpy_message_converter_msgs__msg__TestArray;

// Struct for a sequence of rclpy_message_converter_msgs__msg__TestArray.
typedef struct rclpy_message_converter_msgs__msg__TestArray__Sequence
{
  rclpy_message_converter_msgs__msg__TestArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__msg__TestArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_H_
