from setuptools import find_packages, setup
import os 
from glob import glob
package_name = 'usar_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arms',
    maintainer_email='arms@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plot_node = usar_slam.plot_raw_obs:main',
            'node = usar_slam.slam_frontend:main',
            'fm_node = usar_slam.feature_match:main',
        ],
    },
)
