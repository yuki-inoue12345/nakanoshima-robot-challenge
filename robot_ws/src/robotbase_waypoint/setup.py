from setuptools import find_packages, setup


package_name = 'robotbase_waypoint'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kotantu-desktop',
    maintainer_email='Frog7352@icloud.com',
    description='YAML waypoint following and recording tools for Robotbase.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_follow = robotbase_waypoint.waypoint_follow:main',
            'waypoint_record = robotbase_waypoint.waypoint_record:main',
            'waypoint_visualize = robotbase_waypoint.waypoint_visualize:main',
        ],
    },
)
