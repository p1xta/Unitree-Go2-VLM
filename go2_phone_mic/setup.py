from setuptools import find_packages, setup
from glob import glob

package_name = 'go2_phone_mic'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/static', glob('go2_phone_mic/static/*')),
    ],
    install_requires=['setuptools', 'aiohttp', 'numpy'],
    zip_safe=True,
    maintainer='dmitriy',
    maintainer_email='d.malchenkov@g.nsu.ru',
    description='Phone microphone over WebSocket',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'phone_mic_node = go2_phone_mic.phone_mic_node:main',
        ],
    },
    include_package_data=True,
    package_data={'go2_phone_mic': ['static/*']},
)
