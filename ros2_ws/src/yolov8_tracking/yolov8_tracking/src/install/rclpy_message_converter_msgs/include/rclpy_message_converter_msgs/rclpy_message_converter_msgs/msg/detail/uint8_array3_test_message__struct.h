// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from rclpy_message_converter_msgs:msg/Uint8Array3TestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY3_TEST_MESSAGE__STRUCT_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY3_TEST_MESSAGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Uint8Array3TestMessage in the package rclpy_message_converter_msgs.
/**
  * Fixed size uint8 array for testing purposes
 */
typedef struct rclpy_message_converter_msgs__msg__Uint8Array3TestMessage
{
  uint8_t data[3];
} rclpy_message_converter_msgs__msg__Uint8Array3TestMessage;

// Struct for a sequence of rclpy_message_converter_msgs__msg__Uint8Array3TestMessage.
typedef struct rclpy_message_converter_msgs__msg__Uint8Array3TestMessage__Sequence
{
  rclpy_message_converter_msgs__msg__Uint8Array3TestMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__msg__Uint8Array3TestMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY3_TEST_MESSAGE__STRUCT_H_
