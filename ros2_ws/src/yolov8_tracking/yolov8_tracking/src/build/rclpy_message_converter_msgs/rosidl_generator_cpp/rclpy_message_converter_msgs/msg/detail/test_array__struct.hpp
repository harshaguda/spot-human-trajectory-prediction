// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from rclpy_message_converter_msgs:msg/TestArray.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__rclpy_message_converter_msgs__msg__TestArray __attribute__((deprecated))
#else
# define DEPRECATED__rclpy_message_converter_msgs__msg__TestArray __declspec(deprecated)
#endif

namespace rclpy_message_converter_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TestArray_
{
  using Type = TestArray_<ContainerAllocator>;

  explicit TestArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit TestArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _data_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _data_type data;

  // setters for named parameter idiom
  Type & set__data(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->data = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__rclpy_message_converter_msgs__msg__TestArray
    std::shared_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__rclpy_message_converter_msgs__msg__TestArray
    std::shared_ptr<rclpy_message_converter_msgs::msg::TestArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TestArray_ & other) const
  {
    if (this->data != other.data) {
      return false;
    }
    return true;
  }
  bool operator!=(const TestArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TestArray_

// alias to use template instance with default allocator
using TestArray =
  rclpy_message_converter_msgs::msg::TestArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace rclpy_message_converter_msgs

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__MSG__DETAIL__TEST_ARRAY__STRUCT_HPP_
