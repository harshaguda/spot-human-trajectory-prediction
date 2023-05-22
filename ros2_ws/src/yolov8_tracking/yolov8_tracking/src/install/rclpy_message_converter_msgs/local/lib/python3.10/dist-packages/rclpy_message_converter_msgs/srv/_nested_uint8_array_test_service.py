# generated from rosidl_generator_py/resource/_idl.py.em
# with input from rclpy_message_converter_msgs:srv/NestedUint8ArrayTestService.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_NestedUint8ArrayTestService_Request(type):
    """Metaclass of message 'NestedUint8ArrayTestService_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('rclpy_message_converter_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'rclpy_message_converter_msgs.srv.NestedUint8ArrayTestService_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__nested_uint8_array_test_service__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__nested_uint8_array_test_service__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__nested_uint8_array_test_service__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__nested_uint8_array_test_service__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__nested_uint8_array_test_service__request

            from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
            if NestedUint8ArrayTestMessage.__class__._TYPE_SUPPORT is None:
                NestedUint8ArrayTestMessage.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class NestedUint8ArrayTestService_Request(metaclass=Metaclass_NestedUint8ArrayTestService_Request):
    """Message class 'NestedUint8ArrayTestService_Request'."""

    __slots__ = [
        '_input',
    ]

    _fields_and_field_types = {
        'input': 'rclpy_message_converter_msgs/NestedUint8ArrayTestMessage',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['rclpy_message_converter_msgs', 'msg'], 'NestedUint8ArrayTestMessage'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
        self.input = kwargs.get('input', NestedUint8ArrayTestMessage())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.input != other.input:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property  # noqa: A003
    def input(self):  # noqa: A003
        """Message field 'input'."""
        return self._input

    @input.setter  # noqa: A003
    def input(self, value):  # noqa: A003
        if __debug__:
            from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
            assert \
                isinstance(value, NestedUint8ArrayTestMessage), \
                "The 'input' field must be a sub message of type 'NestedUint8ArrayTestMessage'"
        self._input = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_NestedUint8ArrayTestService_Response(type):
    """Metaclass of message 'NestedUint8ArrayTestService_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('rclpy_message_converter_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'rclpy_message_converter_msgs.srv.NestedUint8ArrayTestService_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__nested_uint8_array_test_service__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__nested_uint8_array_test_service__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__nested_uint8_array_test_service__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__nested_uint8_array_test_service__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__nested_uint8_array_test_service__response

            from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
            if NestedUint8ArrayTestMessage.__class__._TYPE_SUPPORT is None:
                NestedUint8ArrayTestMessage.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class NestedUint8ArrayTestService_Response(metaclass=Metaclass_NestedUint8ArrayTestService_Response):
    """Message class 'NestedUint8ArrayTestService_Response'."""

    __slots__ = [
        '_output',
    ]

    _fields_and_field_types = {
        'output': 'rclpy_message_converter_msgs/NestedUint8ArrayTestMessage',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['rclpy_message_converter_msgs', 'msg'], 'NestedUint8ArrayTestMessage'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
        self.output = kwargs.get('output', NestedUint8ArrayTestMessage())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.output != other.output:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def output(self):
        """Message field 'output'."""
        return self._output

    @output.setter
    def output(self, value):
        if __debug__:
            from rclpy_message_converter_msgs.msg import NestedUint8ArrayTestMessage
            assert \
                isinstance(value, NestedUint8ArrayTestMessage), \
                "The 'output' field must be a sub message of type 'NestedUint8ArrayTestMessage'"
        self._output = value


class Metaclass_NestedUint8ArrayTestService(type):
    """Metaclass of service 'NestedUint8ArrayTestService'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('rclpy_message_converter_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'rclpy_message_converter_msgs.srv.NestedUint8ArrayTestService')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__nested_uint8_array_test_service

            from rclpy_message_converter_msgs.srv import _nested_uint8_array_test_service
            if _nested_uint8_array_test_service.Metaclass_NestedUint8ArrayTestService_Request._TYPE_SUPPORT is None:
                _nested_uint8_array_test_service.Metaclass_NestedUint8ArrayTestService_Request.__import_type_support__()
            if _nested_uint8_array_test_service.Metaclass_NestedUint8ArrayTestService_Response._TYPE_SUPPORT is None:
                _nested_uint8_array_test_service.Metaclass_NestedUint8ArrayTestService_Response.__import_type_support__()


class NestedUint8ArrayTestService(metaclass=Metaclass_NestedUint8ArrayTestService):
    from rclpy_message_converter_msgs.srv._nested_uint8_array_test_service import NestedUint8ArrayTestService_Request as Request
    from rclpy_message_converter_msgs.srv._nested_uint8_array_test_service import NestedUint8ArrayTestService_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
