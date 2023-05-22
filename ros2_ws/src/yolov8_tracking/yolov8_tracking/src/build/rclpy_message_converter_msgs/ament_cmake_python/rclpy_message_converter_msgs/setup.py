from setuptools import find_packages
from setuptools import setup

setup(
    name='rclpy_message_converter_msgs',
    version='2.0.1',
    packages=find_packages(
        include=('rclpy_message_converter_msgs', 'rclpy_message_converter_msgs.*')),
)
