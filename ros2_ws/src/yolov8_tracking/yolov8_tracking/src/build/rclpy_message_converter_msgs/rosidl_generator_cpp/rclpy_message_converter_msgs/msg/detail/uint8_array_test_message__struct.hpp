// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from rclpy_message_converter_msgs:msg/Uint8ArrayTestMessage.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage __attribute__((deprecated))
#else
# define DEPRECATED__rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage __declspec(deprecated)
#endif

namespace rclpy_message_converter_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Uint8ArrayTestMessage_
{
  using Type = Uint8ArrayTestMessage_<ContainerAllocator>;

  explicit Uint8ArrayTestMessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit Uint8ArrayTestMessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _data_type =
    std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>>;
  _data_type data;

  // setters for named parameter idiom
  Type & set__data(
    const std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>> & _arg)
  {
    this->data = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage
    std::shared_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__rclpy_message_converter_msgs__msg__Uint8ArrayTestMessage
    std::shared_ptr<rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Uint8ArrayTestMessage_ & other) const
  {
    if (this->data != other.data) {
      return false;
    }
    return true;
  }
  bool operator!=(const Uint8ArrayTestMessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Uint8ArrayTestMessage_

// alias to use template instance with default allocator
using Uint8ArrayTestMessage =
  rclpy_message_converter_msgs::msg::Uint8ArrayTestMessage_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace rclpy_message_converter_msgs

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__UINT8_ARRAY_TEST_MESSAGE__STRUCT_HPP_
