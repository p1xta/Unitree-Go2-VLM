import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'go2_speech_recognition'

vosk_model_dir = 'weights/vosk-model-small-ru-0.22'
vosk_data_files = []
for dirpath, dirnames, filenames in os.walk(vosk_model_dir):
    if filenames:
        install_dir = os.path.join('share', package_name, dirpath)
        vosk_data_files.append((install_dir, [os.path.join(dirpath, f) for f in filenames]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'weights'), ['weights/best_model.pth']),
    ] + vosk_data_files,
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
            'speech_recognition_node = go2_speech_recognition.speech_recognition_node:main',
        ],
    },
)
