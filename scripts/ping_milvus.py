"""
检测 pymilvus 能否连上项目 .env 中的 MILVUS_URI（load_dotenv(..., override=True)）。

说明（重要）：
  pymilvus 报错「Fail connecting to server ... illegal connection params or server unavailable」
  在源码里对应的是 **gRPC channel 在 timeout 内未变为 ready**（多为 ``grpc.FutureTimeoutError``），
  与「TCP 端口能 telnet」不是同一层：TCP 通只说明有进程在监听，**HTTP/2（gRPC）握手**仍可能失败。

常见原因：
  · 系统/IDE/Clash 等设置了 HTTP(S)_PROXY，干扰本机 gRPC 连局域网 IP → 设 MILVUS_DISABLE_PROXY=1 或手动 unset 代理
  · MILVUS_TIMEOUT 过短 → 默认已改为 90 秒，可在 .env 设 MILVUS_TIMEOUT=120
  · Milvus 容器未 healthy / 依赖 etcd、minio 未起 → 虚拟机内 docker logs、curl 健康检查
  · TCP 探测即失败（timed out / 拒绝连接）→ 与 gRPC/HTTP 代理无关；查网段、防火墙、Milvus 是否监听、IP 是否变更

用法：python milvus_ping.py
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

# 与 _maybe_strip_proxy / 自动重试共用：这些变量会让部分环境下的 grpcio 走错路径，局域网 Milvus 常需 unset
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _proxy_env_is_set() -> bool:
    return any(os.environ.get(k) for k in _PROXY_ENV_KEYS)


def _clear_proxy_env() -> None:
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def _parse_milvus_host_port(uri: str) -> tuple[str, int]:
    raw = uri.strip()
    if "://" not in raw:
        raw = "http://" + raw
    u = urlparse(raw)
    host = u.hostname
    port = u.port
    if not host:
        raise ValueError(f"无法从 MILVUS_URI 解析主机: {uri!r}")
    if port is None:
        port = 19530
    return host, port


def _tcp_ok(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as e:
        return False, str(e)


def _print_proxy_hint() -> None:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    found = [(k, os.environ.get(k, "")) for k in keys if os.environ.get(k)]
    if not found:
        print("代理相关环境变量: (未设置)")
        return
    print("代理相关环境变量（若连局域网 Milvus 失败，可 unset 或设 MILVUS_DISABLE_PROXY=1）:")
    for k, v in found:
        show = v if len(v) < 120 else v[:117] + "..."
        print(f"  {k}={show!r}")


def _maybe_strip_proxy() -> None:
    raw = os.environ.get("MILVUS_DISABLE_PROXY", "").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return
    _clear_proxy_env()
    print("已按 MILVUS_DISABLE_PROXY 清除 HTTP(S)/ALL_PROXY（本进程内）。")


def _raw_grpc_channel_ready(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """不经过 pymilvus，直接测 gRPC channel 是否 ready。"""
    try:
        import grpc
    except ImportError:
        return False, "未安装 grpcio（pymilvus 依赖）"

    target = f"{host}:{port}"
    # grpcio 要求 options 为 (key, value) 元组列表，不能传 dict（否则会按 key 迭代并解包失败）
    ch = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    try:
        grpc.channel_ready_future(ch).result(timeout=timeout)
        return True, ""
    except Exception as e:
        return False, repr(e)
    finally:
        try:
            ch.close()
        except Exception:
            pass


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("需要: pip install python-dotenv", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=True)
    _maybe_strip_proxy()

    uri = (
        os.environ.get("MILVUS_URI")
        or os.environ.get("MILVUS_URL")
        or "http://127.0.0.1:19530"
    ).strip()
    token = (os.environ.get("MILVUS_TOKEN") or "").strip() or None
    raw_t = (os.environ.get("MILVUS_TIMEOUT") or "90").strip()
    try:
        timeout = float(raw_t)
    except ValueError:
        timeout = 90.0

    raw_tcp = (os.environ.get("MILVUS_TCP_TIMEOUT") or "5").strip()
    try:
        tcp_timeout = float(raw_tcp)
    except ValueError:
        tcp_timeout = 5.0
    tcp_timeout = max(0.5, min(tcp_timeout, 120.0))

    print(f"MILVUS_URI={uri}")
    print(f"MILVUS_TIMEOUT={timeout}（gRPC channel ready 等待上限，秒）")
    print(f"MILVUS_TCP_TIMEOUT={tcp_timeout}（仅 TCP 连通性探测，秒；可在 .env 调整）")
    _print_proxy_hint()

    try:
        host, port = _parse_milvus_host_port(uri)
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 1

    ok, err = _tcp_ok(host, port, timeout=tcp_timeout)
    print(f"TCP {host}:{port} → {'可达' if ok else '不可达'}")
    if not ok:
        print(f"  原因: {err}", file=sys.stderr)
        print(
            "  说明: 这一步是操作系统原生 TCP，**不受** HTTP(S)_PROXY 影响；timed out 多为路由/防火墙/服务未监听或 IP 已变。\n"
            "  建议: ping 目标机；PowerShell: Test-NetConnection <host> -Port 19530；"
            "在跑 Milvus 的机器上检查 docker ps / 端口是否在听（Linux: ss -lntp；Windows: netstat -ano | findstr 19530）；"
            "核对 MILVUS_URI 里的 IP 是否仍为当前地址；检查两端与路由器防火墙是否放行 19530。",
            file=sys.stderr,
        )
        return 1

    gr_timeout = min(timeout, 120.0)
    gr_ok, gr_msg = _raw_grpc_channel_ready(host, port, timeout=gr_timeout)
    grpc_retried_clear_proxy = False
    if not gr_ok and _proxy_env_is_set():
        print(
            "裸 gRPC 首次失败且检测到 HTTP(S)/ALL_PROXY："
            "部分环境下 grpcio 连局域网会被代理干扰，本进程内清除这些变量并重试一次…",
            file=sys.stderr,
        )
        _clear_proxy_env()
        grpc_retried_clear_proxy = True
        gr_ok, gr_msg = _raw_grpc_channel_ready(host, port, timeout=gr_timeout)

    grpc_note = ""
    if grpc_retried_clear_proxy and gr_ok:
        grpc_note = "（清除代理后重试成功；后续 MilvusClient 同在本进程无代理环境）"
    print(f"裸 gRPC channel_ready（grpcio）→ {'成功' if gr_ok else '失败'}{grpc_note}")
    if not gr_ok:
        print(
            f"  详情: {gr_msg}\n"
            "  说明: TCP 已通但 gRPC 未就绪，通常不是「密码错误」，而是代理/防火墙/Milvus 未就绪。\n"
            "  建议在虚拟机内执行: python -c \"from pymilvus import MilvusClient; "
            "print(MilvusClient(uri='http://127.0.0.1:19530',timeout=30).list_collections())\"\n"
            "  若仅在 Windows 失败、虚拟机本机成功：在 .env 设 MILVUS_DISABLE_PROXY=1，或为该主机配置 NO_PROXY。",
            file=sys.stderr,
        )

    try:
        import pymilvus
        from pymilvus import MilvusClient

        print(f"pymilvus 版本: {pymilvus.__version__}")
    except ImportError:
        print("需要: pip install pymilvus", file=sys.stderr)
        return 1

    kw: dict = {"uri": uri, "timeout": timeout}
    if token:
        kw["token"] = token

    try:
        print("MilvusClient.list_collections() …")
        client = MilvusClient(**kw)
        names = client.list_collections()
        print(f"连接成功。当前集合数量: {len(names)}")
        return 0
    except Exception as e:
        print(f"MilvusClient 失败: {e}", file=sys.stderr)
        print(
            "可尝试: 1) .env 设 MILVUS_DISABLE_PROXY=1 后再运行本脚本\n"
            "       2) MILVUS_TIMEOUT=120\n"
            "       3) 虚拟机 docker ps 确认 milvus-standalone 为 healthy；docker logs milvus-standalone",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
