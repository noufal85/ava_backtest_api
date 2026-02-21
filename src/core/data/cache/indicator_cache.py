"""Redis-backed indicator cache using MessagePack serialisation."""
import hashlib
import json

import msgpack
import redis.asyncio as redis


class IndicatorCache:

    TTL = 7 * 24 * 3600  # 1 week

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)

    def _key(self, market: str, symbol: str, tf: str, name: str, config: dict) -> str:
        h = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        return f"indicator:{market}:{symbol}:{tf}:{name}:{h}"

    async def get(self, market: str, symbol: str, tf: str, name: str, config: dict):
        raw = await self.client.get(self._key(market, symbol, tf, name, config))
        return msgpack.unpackb(raw) if raw else None

    async def put(self, value, market: str, symbol: str, tf: str, name: str, config: dict) -> None:
        await self.client.setex(
            self._key(market, symbol, tf, name, config),
            self.TTL,
            msgpack.packb(value, use_bin_type=True),
        )

    async def close(self) -> None:
        await self.client.aclose()
