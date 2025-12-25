from setuptools import setup

package_name = 'cognitive_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jack Amichai',
    maintainer_email='jack@example.com',
    description='Bridge between cognitive service and ROS 2 Nav2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = cognitive_bridge.bridge_node:main',
        ],
    },
)
