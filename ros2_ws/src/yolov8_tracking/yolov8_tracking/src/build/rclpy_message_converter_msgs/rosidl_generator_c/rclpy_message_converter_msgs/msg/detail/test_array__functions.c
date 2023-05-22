// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from rclpy_message_converter_msgs:msg/TestArray.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/msg/detail/test_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
rclpy_message_converter_msgs__msg__TestArray__init(rclpy_message_converter_msgs__msg__TestArray * msg)
{
  if (!msg) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__double__Sequence__init(&msg->data, 0)) {
    rclpy_message_converter_msgs__msg__TestArray__fini(msg);
    return false;
  }
  return true;
}

void
rclpy_message_converter_msgs__msg__TestArray__fini(rclpy_message_converter_msgs__msg__TestArray * msg)
{
  if (!msg) {
    return;
  }
  // data
  rosidl_runtime_c__double__Sequence__fini(&msg->data);
}

bool
rclpy_message_converter_msgs__msg__TestArray__are_equal(const rclpy_message_converter_msgs__msg__TestArray * lhs, const rclpy_message_converter_msgs__msg__TestArray * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->data), &(rhs->data)))
  {
    return false;
  }
  return true;
}

bool
rclpy_message_converter_msgs__msg__TestArray__copy(
  const rclpy_message_converter_msgs__msg__TestArray * input,
  rclpy_message_converter_msgs__msg__TestArray * output)
{
  if (!input || !output) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->data), &(output->data)))
  {
    return false;
  }
  return true;
}

rclpy_message_converter_msgs__msg__TestArray *
rclpy_message_converter_msgs__msg__TestArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__TestArray * msg = (rclpy_message_converter_msgs__msg__TestArray *)allocator.allocate(sizeof(rclpy_message_converter_msgs__msg__TestArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(rclpy_message_converter_msgs__msg__TestArray));
  bool success = rclpy_message_converter_msgs__msg__TestArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
rclpy_message_converter_msgs__msg__TestArray__destroy(rclpy_message_converter_msgs__msg__TestArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    rclpy_message_converter_msgs__msg__TestArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
rclpy_message_converter_msgs__msg__TestArray__Sequence__init(rclpy_message_converter_msgs__msg__TestArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__TestArray * data = NULL;

  if (size) {
    data = (rclpy_message_converter_msgs__msg__TestArray *)allocator.zero_allocate(size, sizeof(rclpy_message_converter_msgs__msg__TestArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = rclpy_message_converter_msgs__msg__TestArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        rclpy_message_converter_msgs__msg__TestArray__fini(&data[i - 1]);
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
rclpy_message_converter_msgs__msg__TestArray__Sequence__fini(rclpy_message_converter_msgs__msg__TestArray__Sequence * array)
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
      rclpy_message_converter_msgs__msg__TestArray__fini(&array->data[i]);
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

rclpy_message_converter_msgs__msg__TestArray__Sequence *
rclpy_message_converter_msgs__msg__TestArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__msg__TestArray__Sequence * array = (rclpy_message_converter_msgs__msg__TestArray__Sequence *)allocator.allocate(sizeof(rclpy_message_converter_msgs__msg__TestArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = rclpy_message_converter_msgs__msg__TestArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
rclpy_message_converter_msgs__msg__TestArray__Sequence__destroy(rclpy_message_converter_msgs__msg__TestArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    rclpy_message_converter_msgs__msg__TestArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
rclpy_message_converter_msgs__msg__TestArray__Sequence__are_equal(const rclpy_message_converter_msgs__msg__TestArray__Sequence * lhs, const rclpy_message_converter_msgs__msg__TestArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!rclpy_message_converter_msgs__msg__TestArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
rclpy_message_converter_msgs__msg__TestArray__Sequence__copy(
  const rclpy_message_converter_msgs__msg__TestArray__Sequence * input,
  rclpy_message_converter_msgs__msg__TestArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(rclpy_message_converter_msgs__msg__TestArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rclpy_message_converter_msgs__msg__TestArray * data =
      (rclpy_message_converter_msgs__msg__TestArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!rclpy_message_converter_msgs__msg__TestArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          rclpy_message_converter_msgs__msg__TestArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!rclpy_message_converter_msgs__msg__TestArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
