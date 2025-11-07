import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _convert_msgpack_to_numpy(obj):
    """递归地将msgpack格式的numpy数组字典转换为真正的numpy数组
    
    msgpack格式: {b'data': bytes, b'shape': tuple, b'type': str, b'kind': str, b'nd': bool}
    转换为: numpy.ndarray
    """
    if isinstance(obj, dict):
        # 检查是否是msgpack格式的numpy数组
        if b'data' in obj and b'shape' in obj and b'type' in obj:
            try:
                array_bytes = obj[b'data']
                shape = tuple(obj[b'shape'])
                dtype_str = obj[b'type'].decode() if isinstance(obj[b'type'], bytes) else obj[b'type']
                array = np.frombuffer(array_bytes, dtype=dtype_str).reshape(shape)
                logger.debug(f"转换msgpack数组: shape={shape}, dtype={dtype_str}")
                return array
            except Exception as e:
                logger.error(f"转换msgpack数组失败: {e}")
                raise
        else:
            # 递归处理字典中的每个值，同时将字节串键转换为字符串键
            return {
                (k.decode() if isinstance(k, bytes) else k): _convert_msgpack_to_numpy(v)
                for k, v in obj.items()
            }
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表/元组
        return type(obj)(_convert_msgpack_to_numpy(item) for item in obj)
    else:
        # 其他类型直接返回
        return obj


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw_data = await websocket.recv()
                logger.info(f"[Server] 接收到原始数据大小: {len(raw_data)} 字节")
                obs = msgpack_numpy.unpackb(raw_data)
                logger.info(f"[Server] 反序列化后的obs keys: {obs.keys()}")
                
                # 转换msgpack格式的numpy数组为真正的numpy数组
                obs = _convert_msgpack_to_numpy(obs)
                logger.info(f"[Server] 转换后的obs keys: {obs.keys()}")
                
                for key, value in obs.items():
                    if isinstance(value, dict):
                        logger.info(f"[Server] obs['{key}'] = dict with keys: {value.keys()}")
                    else:
                        logger.info(f"[Server] obs['{key}'] = {type(value)}, shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', 'N/A')}")

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                logger.info(f"[Server] 推理完成，耗时: {infer_time*1000:.2f}ms")
                logger.info(f"[Server] 返回的action keys: {action.keys()}")
                for key, value in action.items():
                    if hasattr(value, 'shape'):
                        logger.info(f"[Server] action['{key}'] shape: {value.shape}, dtype: {value.dtype}")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                packed_action = packer.pack(action)
                logger.info(f"[Server] 发送响应数据，大小: {len(packed_action)} 字节")
                await websocket.send(packed_action)
                logger.info(f"[Server] 响应发送成功")
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
