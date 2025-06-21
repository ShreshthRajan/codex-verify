# src/orchestration/caching_layer.py
"""
Caching Layer - Performance optimization through intelligent caching.
Provides LRU cache, disk-based caching, and cache invalidation strategies.
"""

import asyncio
import time
import pickle
import hashlib
import os
import tempfile
import json
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import lru_cache
import threading
import weakref


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: float
    hit_count: int = 0
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    disk_usage_bytes: int = 0
    avg_retrieval_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate


class CachingLayer:
    """
    High-performance caching layer with multiple storage tiers.
    
    Features:
    - In-memory LRU cache for frequently accessed items
    - Disk-based cache for larger items and persistence
    - Automatic cache invalidation and cleanup
    - Performance monitoring and statistics
    - Thread-safe operations
    - Cache warming and preloading
    """
    
    def __init__(self, enabled: bool = True, ttl: int = 3600, 
                 memory_limit_mb: int = 100, disk_limit_mb: int = 500):
        self.enabled = enabled
        self.default_ttl = ttl
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_limit_bytes = disk_limit_mb * 1024 * 1024
        
        # In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_usage = 0
        
        # Disk cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "codex_verify_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_expired()
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            pass  # No event loop running
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            with self._lock:
                # Check memory cache first
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    if not entry.is_expired:
                        entry.hit_count += 1
                        self.stats.cache_hits += 1
                        self._update_avg_retrieval_time(time.time() - start_time)
                        return entry.value
                    else:
                        # Remove expired entry
                        del self._memory_cache[key]
                        self._memory_usage -= entry.size_bytes
                
                # Check disk cache
                disk_value = await self._get_from_disk(key)
                if disk_value is not None:
                    # Promote to memory cache if small enough
                    value_size = self._estimate_size(disk_value)
                    if value_size < self.memory_limit_bytes // 4:  # Only if <25% of memory limit
                        await self._put_memory(key, disk_value, self.default_ttl)
                    
                    self.stats.cache_hits += 1
                    self._update_avg_retrieval_time(time.time() - start_time)
                    return disk_value
                
                # Cache miss
                self.stats.cache_misses += 1
                return None
                
        except Exception:
            self.stats.cache_misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successfully cached
        """
        if not self.enabled:
            return False
        
        ttl = ttl or self.default_ttl
        value_size = self._estimate_size(value)
        
        try:
            with self._lock:
                # Decide storage tier based on size
                if value_size < self.memory_limit_bytes // 10:  # Small items go to memory
                    return await self._put_memory(key, value, ttl)
                else:  # Large items go to disk
                    return await self._put_disk(key, value, ttl)
                    
        except Exception:
            return False
    
    async def _put_memory(self, key: str, value: Any, ttl: int) -> bool:
        """Store item in memory cache"""
        value_size = self._estimate_size(value)
        
        # Evict items if necessary
        while (self._memory_usage + value_size) > self.memory_limit_bytes and self._memory_cache:
            await self._evict_lru_memory()
        
        # Store new entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            size_bytes=value_size
        )
        
        # Remove old entry if exists
        if key in self._memory_cache:
            self._memory_usage -= self._memory_cache[key].size_bytes
        
        self._memory_cache[key] = entry
        self._memory_usage += value_size
        
        return True
    
    async def _put_disk(self, key: str, value: Any, ttl: int) -> bool:
        """Store item in disk cache"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            # Create cache entry metadata
            entry_data = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl,
                'key': key
            }
            
            # Write to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            # Check disk usage and cleanup if necessary
            await self._cleanup_disk_if_needed()
            
            return True
            
        except Exception:
            return False
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve item from disk cache"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if not cache_file.exists():
                return None
            
            # Load from disk
            with open(cache_file, 'rb') as f:
                entry_data = pickle.load(f)
            
            # Check expiration
            if time.time() > (entry_data['timestamp'] + entry_data['ttl']):
                # Remove expired file
                cache_file.unlink(missing_ok=True)
                return None
            
            return entry_data['value']
            
        except Exception:
            return None
    
    async def _evict_lru_memory(self):
        """Evict least recently used item from memory cache"""
        if not self._memory_cache:
            return
        
        # Find LRU item (oldest timestamp, lowest hit count)
        lru_key = min(self._memory_cache.keys(), 
                     key=lambda k: (self._memory_cache[k].hit_count, self._memory_cache[k].timestamp))
        
        entry = self._memory_cache.pop(lru_key)
        self._memory_usage -= entry.size_bytes
        self.stats.evictions += 1
        
        # Optionally write to disk if valuable
        if entry.hit_count > 2:  # Item was accessed multiple times
            await self._put_disk(lru_key, entry.value, int(entry.ttl - entry.age_seconds))
    
    async def _cleanup_disk_if_needed(self):
        """Cleanup disk cache if over limit"""
        disk_usage = self._calculate_disk_usage()
        
        if disk_usage > self.disk_limit_bytes:
            # Get all cache files with their ages
            cache_files = []
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    stat = cache_file.stat()
                    cache_files.append((cache_file, stat.st_mtime, stat.st_size))
                except Exception:
                    continue
            
            # Sort by age (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            for cache_file, _, size in cache_files:
                cache_file.unlink(missing_ok=True)
                disk_usage -= size
                if disk_usage <= self.disk_limit_bytes * 0.8:  # Remove extra 20%
                    break
    
    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        
        with self._lock:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if current_time > (entry.timestamp + entry.ttl)
            ]
            
            for key in expired_keys:
                entry = self._memory_cache.pop(key)
                self._memory_usage -= entry.size_bytes
        
        # Clean disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry_data = pickle.load(f)
                
                if current_time > (entry_data['timestamp'] + entry_data['ttl']):
                    cache_file.unlink(missing_ok=True)
                    
            except Exception:
                # Remove corrupted files
                cache_file.unlink(missing_ok=True)
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path for key"""
        # Create safe filename from key
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                return 1024  # Default estimate
    
    def _calculate_disk_usage(self) -> int:
        """Calculate total disk cache usage"""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                total_size += cache_file.stat().st_size
            except Exception:
                continue
        return total_size
    
    def _update_avg_retrieval_time(self, retrieval_time: float):
        """Update average retrieval time statistics"""
        if self.stats.avg_retrieval_time == 0:
            self.stats.avg_retrieval_time = retrieval_time
        else:
            # Exponential moving average
            self.stats.avg_retrieval_time = (0.9 * self.stats.avg_retrieval_time + 
                                           0.1 * retrieval_time)
    
    async def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._memory_cache.clear()
            self._memory_usage = 0
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)
        
        # Reset stats
        self.stats = CacheStats()
    
    async def delete(self, key: str) -> bool:
        """Delete specific cache entry"""
        deleted = False
        
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._memory_usage -= entry.size_bytes
                deleted = True
        
        # Delete from disk
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            cache_file.unlink(missing_ok=True)
            deleted = True
        
        return deleted
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        stats = CacheStats(
            total_requests=self.stats.total_requests,
            cache_hits=self.stats.cache_hits,
            cache_misses=self.stats.cache_misses,
            evictions=self.stats.evictions,
            memory_usage_bytes=self._memory_usage,
            disk_usage_bytes=self._calculate_disk_usage(),
            avg_retrieval_time=self.stats.avg_retrieval_time
        )
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        
        try:
            # Test memory cache
            test_key = "health_check"
            test_value = {"test": True, "timestamp": time.time()}
            
            start_time = time.time()
            await self.set(test_key, test_value)
            cached_value = await self.get(test_key)
            retrieval_time = time.time() - start_time
            
            memory_ok = cached_value is not None
            
            # Check disk cache
            disk_ok = self.cache_dir.exists() and self.cache_dir.is_dir()
            
            # Calculate cache usage
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_disk_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            # Determine status
            status = "healthy"
            warnings = []
            
            if not memory_ok:
                status = "degraded"
                warnings.append("Memory cache test failed")
            
            if total_disk_size > 100 * 1024 * 1024:  # 100MB warning
                warnings.append("High disk cache usage")
            
            if retrieval_time > 0.1:  # 100ms warning
                warnings.append("Slow cache retrieval")
            
            return {
                "status": status,
                "memory_cache_entries": len(self.memory_cache),
                "disk_cache_entries": len(cache_files),
                "disk_usage_mb": total_disk_size / (1024 * 1024),
                "cache_directory": str(self.cache_dir),
                "ttl": self.ttl,
                "retrieval_time_ms": retrieval_time * 1000,
                "warnings": warnings if warnings else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
        
    async def warm_cache(self, items: List[Tuple[str, Any, int]]):
        """Pre-warm cache with provided items"""
        for key, value, ttl in items:
            await self.set(key, value, ttl)
    
    async def cleanup_with_size_management(self):
        """Enhanced cleanup with intelligent size management"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Cleanup memory cache
        expired_keys = [
            key for key, (_, timestamp) in self.memory_cache.items()
            if current_time - timestamp >= self.ttl
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Enhanced disk cleanup with size management
        cache_files = []
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                stat = cache_file.stat()
                
                # Check if expired
                with open(cache_file, 'rb') as f:
                    cached_data, timestamp = pickle.load(f)
                
                if current_time - timestamp >= self.ttl:
                    cache_file.unlink()
                else:
                    cache_files.append((cache_file, stat.st_size, timestamp))
                    total_size += stat.st_size
                    
            except Exception:
                # Remove corrupted files
                cache_file.unlink(missing_ok=True)
        
        # If total size > 50MB, remove oldest files
        if total_size > 50 * 1024 * 1024:  # 50MB limit
            # Sort by timestamp (oldest first)
            cache_files.sort(key=lambda x: x[2])
            
            # Remove oldest files until under 40MB
            for cache_file, file_size, _ in cache_files:
                if total_size <= 40 * 1024 * 1024:
                    break
                try:
                    cache_file.unlink()
                    total_size -= file_size
                except Exception:
                    pass
                
    async def cleanup(self):
        """Cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self._cleanup_expired()


# Utility functions for cache management
@lru_cache(maxsize=1000)
def _cached_hash(content: str) -> str:
    """Cached hash function for frequently hashed content"""
    return hashlib.sha256(content.encode()).hexdigest()


def create_cache_key(*args, **kwargs) -> str:
    """Create standardized cache key from arguments"""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, str):
            key_parts.append(arg)
        else:
            key_parts.append(str(hash(str(arg))))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={value}")
    
    key_string = "|".join(key_parts)
    return _cached_hash(key_string)