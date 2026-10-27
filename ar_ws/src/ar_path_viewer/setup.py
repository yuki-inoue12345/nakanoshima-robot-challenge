from glob import glob
from setuptools import find_packages, setup


package_name = "ar_path_viewer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/test_data", glob("test_data/*")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Koko-chan AR project",
    maintainer_email="maintainer@example.com",
    description="Read-only AR overlay viewer for a Nav2 Path",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ar_path_viewer = ar_path_viewer.viewer_node:main",
            "simulate_lap = ar_path_viewer.simulation:main",
        ],
    },
)
