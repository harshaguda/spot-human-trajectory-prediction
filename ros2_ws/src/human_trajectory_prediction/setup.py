from setuptools import setup
import os
from glob import glob


package_name = 'human_trajectory_prediction'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abdelrahman',
    maintainer_email='abdelrahman@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "prediction_node = human_trajectory_prediction.prediction_node:main",
            "danger_zone_publisher = human_trajectory_prediction.danger_zone_publisher:main"
        ],
    },
)
