// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from rclpy_message_converter_msgs:msg/Uint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_

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

/// Struct defined in msg/Uint8ArrayTestMessage in the package rclpy_message_converter_msgs.
/**
  * Size-agnostic uint8 array for testing purposes
 */
typedef struct rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage
{
  rosidl_runtime_c__uint8__Sequence data;
} rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage;

// Struct for a sequence of rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage.
typedef struct rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence
{
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_
