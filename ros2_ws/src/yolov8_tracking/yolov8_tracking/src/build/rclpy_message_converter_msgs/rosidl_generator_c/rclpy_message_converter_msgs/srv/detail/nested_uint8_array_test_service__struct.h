// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'input'
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.h"

/// Struct defined in srv/NestedUint8ArrayTestService in the package rclpy_message_converter_msgs.
typedef struct rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request
{
  rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage input;
} rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request;

// Struct for a sequence of rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request.
typedef struct rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence
{
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'output'
// already included above
// #include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.h"

/// Struct defined in srv/NestedUint8ArrayTestService in the package rclpy_message_converter_msgs.
typedef struct rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response
{
  rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage output;
} rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response;

// Struct for a sequence of rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response.
typedef struct rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence
{
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_H_
