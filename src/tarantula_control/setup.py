from setuptools import setup

package_name = 'tarantula_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ang',
    maintainer_email='angli23@yahoo.com',
    description='Control and diagnostic nodes for the Tarantula six-wheel chassis',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gazebo_truth_odometry = tarantula_control.gazebo_truth_odometry:main',
            'motion_control_node = tarantula_control.motion_control_node:main',
            'scan_gate = tarantula_control.scan_gate:main',
        ],
    },
)
