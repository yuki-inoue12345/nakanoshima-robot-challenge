"""Install the isolated reference planner package."""

from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'ar_reference_planner'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}', ['README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='inoue-yuki',
    maintainer_email='inoue-yuki@todo.todo',
    description='Isolated Nav2 reference-path planning for AR navigation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'reference_planner_auto_supervisor = '
            'ar_reference_planner.auto_supervisor:main',
            'legacy_follower_reference_bridge = '
            'ar_reference_planner.legacy_follower_bridge:main',
            'reference_path_request = ar_reference_planner.request_node:main',
            'mock_follower_state = ar_reference_planner.mock_follower_state:main',
            'synthetic_map_publisher = ar_reference_planner.synthetic_map:main',
        ],
    },
)
