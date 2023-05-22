// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice
#include "rclpy_message_converter_msgs/srv/detail/nested_uint8_array_test_service__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `input`
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__functions.h"

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg)
{
  if (!msg) {
    return false;
  }
  // input
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__init(&msg->input)) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(msg);
    return false;
  }
  return true;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg)
{
  if (!msg) {
    return;
  }
  // input
  rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__fini(&msg->input);
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // input
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__are_equal(
      &(lhs->input), &(rhs->input)))
  {
    return false;
  }
  return true;
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // input
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__copy(
      &(input->input), &(output->input)))
  {
    return false;
  }
  return true;
}

rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request *)allocator.allocate(sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request));
  bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * data = NULL;

  if (size) {
    data = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request *)allocator.zero_allocate(size, sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(&data[i - 1]);
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array)
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
      rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(&array->data[i]);
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

rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence *)allocator.allocate(sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request * data =
      (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `output`
// already included above
// #include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__functions.h"

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg)
{
  if (!msg) {
    return false;
  }
  // output
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__init(&msg->output)) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(msg);
    return false;
  }
  return true;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg)
{
  if (!msg) {
    return;
  }
  // output
  rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__fini(&msg->output);
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // output
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__are_equal(
      &(lhs->output), &(rhs->output)))
  {
    return false;
  }
  return true;
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // output
  if (!rclpy_message_converter_msgs__msg__NestedUint8ArrayTestMessage__copy(
      &(input->output), &(output->output)))
  {
    return false;
  }
  return true;
}

rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response *)allocator.allocate(sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response));
  bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__init(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * data = NULL;

  if (size) {
    data = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response *)allocator.zero_allocate(size, sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(&data[i - 1]);
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
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__fini(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array)
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
      rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(&array->data[i]);
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

rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence *
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array = (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence *)allocator.allocate(sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__destroy(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__are_equal(const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * lhs, const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence__copy(
  const rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * input,
  rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response * data =
      (rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
