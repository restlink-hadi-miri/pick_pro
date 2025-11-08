from setuptools import setup
from glob import glob
import os

package_name = 'o_detector'

setup(
    name=package_name,
    version='1.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Object detector and picker for UR10e',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_detector = scripts.ball_detector:main',
            'main_picker = scripts.main_picker:main',
            'live_tcp_logger = scripts.live_tcp_logger:main',
        ],
    },
)

