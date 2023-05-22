// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from rclpy_message_converter_msgs:msg/Uint8ArrayTestMessage.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/msg/detail/uint8_array_test_message__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg)
{
  if (!msg) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__uint8__Sequence__init(&msg->data, 0)) {
    rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(msg);
    return false;
  }
  return true;
}

void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg)
{
  if (!msg) {
    return;
  }
  // data
  rosidl_runtime_c__uint8__Sequence__fini(&msg->data);
}

bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__are_equal(const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * lhs, const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__uint8__Sequence__are_equal(
      &(lhs->data), &(rhs->data)))
  {
    return false;
  }
  return true;
}

bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__copy(
  const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * input,
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * output)
{
  if (!input || !output) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__uint8__Sequence__copy(
      &(input->data), &(output->data)))
  {
    return false;
  }
  return true;
}

rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage *
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg = (rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage *)allocator.allocate(sizeof(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage));
  bool success = rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__destroy(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__init(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * data = NULL;

  if (size) {
    data = (rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage *)allocator.zero_allocate(size, sizeof(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__fini(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence *
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array = (rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence *)allocator.allocate(sizeof(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__destroy(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__are_equal(const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * lhs, const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence__copy(
  const rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * input,
  rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage * data =
      (rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
