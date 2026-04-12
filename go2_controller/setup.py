from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'go2_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dmitriy',
    maintainer_email='d.malchenkov@g.nsu.ru',
    description='Controller node: routes VLM commands to /cmd_vel_out (Twist) and /webrtc_req (WebRtcReq)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'go2_controller_node = go2_controller.go2_controller_node:main',
            'move_service_node = go2_controller.move_service_node:main',
        ],
    },
)
