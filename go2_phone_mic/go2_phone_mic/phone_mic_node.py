import asyncio
import os
import ssl
import subprocess
import threading
from pathlib import Path

import numpy as np
import rclpy
from aiohttp import web, WSMsgType
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import socket

from go2_interfaces.msg import AudioData


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512


class PhoneMicNode(Node):
    def __init__(self):
        super().__init__('phone_mic_node')

        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8443)
        self.declare_parameter('cert_dir', os.path.expanduser('~/.go2_phone_mic'))
        self.declare_parameter('audio_topic', '/audio_raw')

        self.host = self.get_parameter('host').value
        self.port = int(self.get_parameter('port').value)
        self.cert_dir = self.get_parameter('cert_dir').value
        topic = self.get_parameter('audio_topic').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.audio_pub = self.create_publisher(AudioData, topic, qos)

        share = get_package_share_directory('go2_phone_mic')
        self.static_dir = Path(share) / 'static'

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._buffer = np.empty(0, dtype=np.int16)

        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5)

    def _run_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = web.Application()
        app.router.add_get('/', self._index)
        app.router.add_get('/ws/audio', self._ws_handler)
        app.router.add_static('/static', str(self.static_dir))

        ssl_context = self._make_ssl_context()

        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, self.host, self.port, ssl_context=ssl_context)
        loop.run_until_complete(site.start())

        self.get_logger().info(
            f'phone mic server: https://<jetson-ip>:{self.port}'
        )
        self._loop_ready.set()
        loop.run_forever()

    async def _index(self, request):
        return web.FileResponse(self.static_dir / 'index.html')

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
        await ws.prepare(request)
        peer = request.remote
        self.get_logger().info(f'phone connected: {peer}')

        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                self._on_pcm(msg.data)
            elif msg.type == WSMsgType.ERROR:
                self.get_logger().warn(f'ws error: {ws.exception()}')

        self.get_logger().info(f'phone disconnected: {peer}')
        return ws

    def _on_pcm(self, raw: bytes) -> None:
        samples = np.frombuffer(raw, dtype=np.int16)
        if samples.size == 0:
            return
        self._buffer = np.concatenate([self._buffer, samples])

        now_us = self.get_clock().now().nanoseconds // 1000
        while len(self._buffer) >= CHUNK_SAMPLES:
            chunk = self._buffer[:CHUNK_SAMPLES]
            self._buffer = self._buffer[CHUNK_SAMPLES:]
            msg = AudioData()
            msg.time_frame = int(now_us)
            msg.data = chunk.tobytes()
            self.audio_pub.publish(msg)

    def _make_ssl_context(self) -> ssl.SSLContext:
        os.makedirs(self.cert_dir, exist_ok=True)
        cert = Path(self.cert_dir) / 'cert.pem'
        key = Path(self.cert_dir) / 'key.pem'
        if not cert.exists() or not key.exists():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            san = f'IP:{local_ip},IP:127.0.0.1,DNS:localhost'
            self.get_logger().info(f'generating self-signed cert for {local_ip}')
            subprocess.run([
                'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                '-keyout', str(key), '-out', str(cert),
                '-days', '3650', '-subj', f'/CN={local_ip}',
                '-addext', f'subjectAltName={san}',
            ], check=True)
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
        return ctx


def main(args=None):
    rclpy.init(args=args)
    node = PhoneMicNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
