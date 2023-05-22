// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__FUNCTIONS_H_
#define RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "rclpy_message_converter_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__struct.h"

/// Initialize srv/NestedUint8ArrayTestService message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request
 * )) before or use
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg);

/// Finalize srv/NestedUint8ArrayTestService message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg);

/// Create srv/NestedUint8ArrayTestService message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__create();

/// Destroy srv/NestedUint8ArrayTestService message.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg);

/// Check for srv/NestedUint8ArrayTestService message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * rhs);

/// Copy a srv/NestedUint8ArrayTestService message.
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * output);

/// Initialize array of srv/NestedUint8ArrayTestService messages.
/**
 * It allocates the memory for the number of elements and calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array, size_t size);

/// Finalize array of srv/NestedUint8ArrayTestService messages.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array);

/// Create array of srv/NestedUint8ArrayTestService messages.
/**
 * It allocates the memory for the array and calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__create(size_t size);

/// Destroy array of srv/NestedUint8ArrayTestService messages.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array);

/// Check for srv/NestedUint8ArrayTestService message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * rhs);

/// Copy an array of srv/NestedUint8ArrayTestService messages.
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * output);

/// Initialize srv/NestedUint8ArrayTestService message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response
 * )) before or use
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg);

/// Finalize srv/NestedUint8ArrayTestService message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg);

/// Create srv/NestedUint8ArrayTestService message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__create();

/// Destroy srv/NestedUint8ArrayTestService message.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg);

/// Check for srv/NestedUint8ArrayTestService message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * rhs);

/// Copy a srv/NestedUint8ArrayTestService message.
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * output);

/// Initialize array of srv/NestedUint8ArrayTestService messages.
/**
 * It allocates the memory for the number of elements and calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array, size_t size);

/// Finalize array of srv/NestedUint8ArrayTestService messages.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array);

/// Create array of srv/NestedUint8ArrayTestService messages.
/**
 * It allocates the memory for the array and calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__create(size_t size);

/// Destroy array of srv/NestedUint8ArrayTestService messages.
/**
 * It calls
 * rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array);

/// Check for srv/NestedUint8ArrayTestService message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_rclpy_message_converter_msgs
bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * rhs);

/// Copy an array of srv/NestedUint8ArrayTestService messages.
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__FUNCTIONS_H_
