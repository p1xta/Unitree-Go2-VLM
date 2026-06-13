import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'go2_speech_generation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'weights'), ['weights/ru_RU-dmitri-medium.onnx']),
        (os.path.join('share', package_name, 'weights'), ['weights/ru_RU-dmitri-medium.onnx.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='p1xta',
    maintainer_email='daria.petrova496@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'speech_playback_node = go2_speech_generation.speech_playback_node:main',
        ],
    },
)
