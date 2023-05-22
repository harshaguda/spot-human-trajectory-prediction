// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from rclpy_message_converter_msgs:msg/Uint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__FUNCTIONS_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "rclpy_message_converter_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "rclpy_message_converter_msgs/msg/detail/uint8_array_test_message__struct.h"

/// Initialize msg/Uint8ArrayTestMessage message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage
 * )) before or use
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg);

/// Finalize msg/Uint8ArrayTestMessage message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg);

/// Create msg/Uint8ArrayTestMessage message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage *
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__create();

/// Destroy msg/Uint8ArrayTestMessage message.
/**
 * It calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__destroy(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg);

/// Check for msg/Uint8ArrayTestMessage message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__are_equal(const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * lhs, const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * rhs);

/// Copy a msg/Uint8ArrayTestMessage message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__copy(
  const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * input,
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * output);

/// Initialize array of msg/Uint8ArrayTestMessage messages.
/**
 * It allocates the memory for the number of elements and calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__init(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array, size_t size);

/// Finalize array of msg/Uint8ArrayTestMessage messages.
/**
 * It calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__fini(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array);

/// Create array of msg/Uint8ArrayTestMessage messages.
/**
 * It allocates the memory for the array and calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence *
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__create(size_t size);

/// Destroy array of msg/Uint8ArrayTestMessage messages.
/**
 * It calls
 * rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__destroy(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array);

/// Check for msg/Uint8ArrayTestMessage message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__are_equal(const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * lhs, const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * rhs);

/// Copy an array of msg/Uint8ArrayTestMessage messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__copy(
  const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * input,
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__FUNCTIONS_H_
