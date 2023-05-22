// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
// generated code does not contain a copyright notice

#ifndef RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_HPP_
#define RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'input'
#include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request __attribute__((deprecated))
#else
# define DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request __declspec(deprecated)
#endif

namespace rclpy_message_converter_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct NestedUint8ArrayTestService_Request_
{
  using Type = NestedUint8ArrayTestService_Request_<ContainerAllocator>;

  explicit NestedUint8ArrayTestService_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : input(_init)
  {
    (void)_init;
  }

  explicit NestedUint8ArrayTestService_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : input(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _input_type =
    rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage_<ContainerAllocator>;
  _input_type input;

  // setters for named parameter idiom
  Type & set__input(
    const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage_<ContainerAllocator> & _arg)
  {
    this->input = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Request
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NestedUint8ArrayTestService_Request_ & other) const
  {
    if (this->input != other.input) {
      return false;
    }
    return true;
  }
  bool operator!=(const NestedUint8ArrayTestService_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NestedUint8ArrayTestService_Request_

// alias to use template instance with default allocator
using NestedUint8ArrayTestService_Request =
  rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace rclpy_message_converter_msgs


// Include directives for member types
// Member 'output'
// already included above
// #include "rclpy_message_converter_msgs/msg/detail/nested_uint8_array_test_message__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response __attribute__((deprecated))
#else
# define DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response __declspec(deprecated)
#endif

namespace rclpy_message_converter_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct NestedUint8ArrayTestService_Response_
{
  using Type = NestedUint8ArrayTestService_Response_<ContainerAllocator>;

  explicit NestedUint8ArrayTestService_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : output(_init)
  {
    (void)_init;
  }

  explicit NestedUint8ArrayTestService_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : output(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _output_type =
    rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage_<ContainerAllocator>;
  _output_type output;

  // setters for named parameter idiom
  Type & set__output(
    const rclpy_message_converter_msgs::msg::NestedUint8ArrayTestMessage_<ContainerAllocator> & _arg)
  {
    this->output = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__rclpy_message_converter_msgs__srv__NestedUint8ArrayTestService_Response
    std::shared_ptr<rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NestedUint8ArrayTestService_Response_ & other) const
  {
    if (this->output != other.output) {
      return false;
    }
    return true;
  }
  bool operator!=(const NestedUint8ArrayTestService_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NestedUint8ArrayTestService_Response_

// alias to use template instance with default allocator
using NestedUint8ArrayTestService_Response =
  rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace rclpy_message_converter_msgs

namespace rclpy_message_converter_msgs
{

namespace srv
{

struct NestedUint8ArrayTestService
{
  using Request = rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Request;
  using Response = rclpy_message_converter_msgs::srv::NestedUint8ArrayTestService_Response;
};

}  // namespace srv

}  // namespace rclpy_message_converter_msgs

#endif  // RCLPY_MESSAGE_CONVERTER_MSGS__SRV__DETAIL__NESTED_UINT8_ARRAY_TEST_SERVICE__STRUCT_HPP_
