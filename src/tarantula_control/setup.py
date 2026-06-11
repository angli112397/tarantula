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
    description='Active suspension controller for the Tarantula six-wheel chassis',
    license='MIT',
    entry_points={
        'console_scripts': [
            'active_suspension = tarantula_control.active_suspension:main',
        ],
    },
)
