// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from rclpy_message_converter_msgs:msg/NestedUint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'arrays'
#include "rclpy_message_converter_msgs/msg/detail/uint8_array_test_message__struct.h"

/// Struct defined in msg/NestedUint8ArrayTestMessage in the package rclpy_message_converter_msgs.
/**
  * array of arrays for testing purposes
 */
typedef struct rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage
{
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence arrays;
} rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage;

// Struct for a sequence of rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage.
typedef struct rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__Sequence
{
  rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__NESTED_UINT8_ARRAY_TEST_MESSAGE__STRUCT_H_
