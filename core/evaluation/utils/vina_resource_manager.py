#!/usr/bin/env python3
"""
Vina Resource Manager for MolPilot
=================================

This module provides SOTA resource management specifically for Vina docking operations
to prevent worker process crashes and resource exhaustion during evaluation.

Key Features:
- Memory-aware Vina instance management
- Process-safe resource allocation
- Automatic cleanup and recovery
- Worker process protection
- Production-ready error handling

Designed specifically for MolPilot's evaluation pipeline with vina_dock mode.
"""

import os
import gc
import time
import psutil
import threading
import weakref
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import tempfile
import shutil

# Global resource tracking
_active_vina_instances = set()  # Changed from WeakSet to regular set
_resource_lock = threading.Lock()
_process_memory_limit = None
_temp_directories = set()  # Changed from WeakSet to regular set for string paths


class VinaResourceManager:
    """
    SOTA resource manager for Vina docking operations.
    """
    
    def __init__(self, memory_limit_mb: Optional[int] = None, max_concurrent_instances: int = 1):
        """
        Initialize Vina resource manager.
        
        Args:
            memory_limit_mb: Memory limit in MB (auto-detect if None)
            max_concurrent_instances: Maximum concurrent Vina instances
        """
        self.memory_limit_mb = memory_limit_mb or self._detect_memory_limit()
        self.max_concurrent_instances = max_concurrent_instances
        self.active_instances = 0
        self.instance_lock = threading.Lock()
        
        # Track temporary files and directories
        self.temp_files = []
        self.temp_dirs = []
        
        print(f"🔧 VinaResourceManager initialized:")
        print(f"   Memory limit: {self.memory_limit_mb} MB")
        print(f"   Max concurrent instances: {self.max_concurrent_instances}")
    
    def _detect_memory_limit(self) -> int:
        """Detect appropriate memory limit based on system resources."""
        try:
            # Get available memory
            memory = psutil.virtual_memory()
            available_mb = memory.available // (1024 * 1024)
            
            # Use 70% of available memory as limit
            limit_mb = int(available_mb * 0.7)
            
            # Minimum 1GB, maximum 8GB per process
            limit_mb = max(1024, min(limit_mb, 8192))
            
            return limit_mb
            
        except Exception as e:
            print(f"⚠️  Memory detection failed: {e}, using default 2GB")
            return 2048
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except Exception as e:
            print(f"⚠️  Memory check failed: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': 0}
    
    def is_memory_available(self, required_mb: int = 500) -> bool:
        """Check if sufficient memory is available."""
        try:
            memory_info = self.check_memory_usage()
            current_mb = memory_info['rss_mb']
            available_mb = memory_info['available_mb']
            
            # Check if we have enough memory
            if current_mb + required_mb > self.memory_limit_mb:
                print(f"⚠️  Memory limit exceeded: {current_mb + required_mb} MB > {self.memory_limit_mb} MB")
                return False
            
            if available_mb < required_mb:
                print(f"⚠️  Insufficient system memory: {available_mb} MB < {required_mb} MB")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  Memory availability check failed: {e}")
            return False
    
    @contextmanager
    def vina_instance_context(self, required_memory_mb: int = 500):
        """
        Context manager for safe Vina instance creation and cleanup.
        
        Args:
            required_memory_mb: Estimated memory requirement for this instance
        """
        instance_acquired = False
        temp_cleanup_list = []
        
        try:
            # Wait for available slot
            with self.instance_lock:
                if self.active_instances >= self.max_concurrent_instances:
                    print(f"⚠️  Max concurrent Vina instances reached: {self.active_instances}")
                    raise RuntimeError("Too many concurrent Vina instances")
                
                # Check memory availability
                if not self.is_memory_available(required_memory_mb):
                    # Try garbage collection
                    print("🧹 Attempting memory cleanup...")
                    collected = gc.collect()
                    print(f"🗑️  Collected {collected} objects")
                    
                    # Check again after cleanup
                    if not self.is_memory_available(required_memory_mb):
                        raise RuntimeError("Insufficient memory for Vina instance")
                
                self.active_instances += 1
                instance_acquired = True
                print(f"✅ Vina instance acquired ({self.active_instances}/{self.max_concurrent_instances})")
            
            # Create temporary directory for this instance
            temp_dir = tempfile.mkdtemp(prefix="vina_instance_")
            temp_cleanup_list.append(temp_dir)
            with _resource_lock:  # Thread-safe access to global set
                _temp_directories.add(temp_dir)
            
            yield temp_dir
            
        except Exception as e:
            print(f"❌ Vina instance context error: {e}")
            raise
            
        finally:
            # Cleanup temporary files and directories
            for temp_path in temp_cleanup_list:
                try:
                    if os.path.exists(temp_path):
                        if os.path.isdir(temp_path):
                            shutil.rmtree(temp_path)
                        else:
                            os.remove(temp_path)
                except Exception as cleanup_error:
                    print(f"⚠️  Temp cleanup failed for {temp_path}: {cleanup_error}")
            
            # Release instance slot
            if instance_acquired:
                with self.instance_lock:
                    self.active_instances -= 1
                    print(f"🔓 Vina instance released ({self.active_instances}/{self.max_concurrent_instances})")
            
            # Force garbage collection
            gc.collect()
    
    def emergency_cleanup(self):
        """Emergency cleanup of all resources."""
        print("🚨 Emergency Vina resource cleanup initiated...")
        
        try:
            # Force garbage collection
            for i in range(3):
                collected = gc.collect()
                if collected > 0:
                    print(f"🗑️  Emergency GC round {i+1}: collected {collected} objects")
            
            # Clean up temporary directories
            temp_dirs_cleaned = 0
            with _resource_lock:  # Thread-safe access to global set
                temp_dirs_to_clean = list(_temp_directories)
                _temp_directories.clear()  # Clear the set after copying

            for temp_dir in temp_dirs_to_clean:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        temp_dirs_cleaned += 1
                except Exception as e:
                    print(f"⚠️  Failed to cleanup temp dir {temp_dir}: {e}")
            
            if temp_dirs_cleaned > 0:
                print(f"🧹 Cleaned up {temp_dirs_cleaned} temporary directories")
            
            # Reset instance counter
            with self.instance_lock:
                self.active_instances = 0
            
            # Print final memory status
            memory_info = self.check_memory_usage()
            print(f"📊 Post-cleanup memory: {memory_info['rss_mb']:.1f} MB RSS, {memory_info['percent']:.1f}%")
            
        except Exception as e:
            print(f"❌ Emergency cleanup error: {e}")
        
        print("✅ Emergency Vina resource cleanup completed")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        memory_info = self.check_memory_usage()
        
        return {
            'active_instances': self.active_instances,
            'max_instances': self.max_concurrent_instances,
            'memory_limit_mb': self.memory_limit_mb,
            'current_memory_mb': memory_info['rss_mb'],
            'memory_percent': memory_info['percent'],
            'available_memory_mb': memory_info['available_mb'],
            'temp_directories': len(_temp_directories)
        }


# Global resource manager instance
_global_vina_resource_manager = None

def get_vina_resource_manager() -> VinaResourceManager:
    """Get or create global Vina resource manager."""
    global _global_vina_resource_manager
    
    if _global_vina_resource_manager is None:
        # Detect if we're in a worker process
        is_worker = hasattr(os, 'getppid') and 'DataLoader' in str(psutil.Process().parent().name())
        
        # Use more conservative settings for worker processes
        max_instances = 1 if is_worker else 2
        
        _global_vina_resource_manager = VinaResourceManager(
            max_concurrent_instances=max_instances
        )
    
    return _global_vina_resource_manager


@contextmanager
def safe_vina_context(required_memory_mb: int = 500):
    """
    Convenience context manager for safe Vina operations.
    
    Args:
        required_memory_mb: Estimated memory requirement
    """
    manager = get_vina_resource_manager()
    with manager.vina_instance_context(required_memory_mb) as temp_dir:
        yield temp_dir


def cleanup_vina_resources():
    """Manual cleanup of Vina resources."""
    global _global_vina_resource_manager
    
    if _global_vina_resource_manager is not None:
        _global_vina_resource_manager.emergency_cleanup()


def print_vina_resource_stats():
    """Print current Vina resource statistics."""
    global _global_vina_resource_manager
    
    if _global_vina_resource_manager is not None:
        stats = _global_vina_resource_manager.get_resource_stats()
        print("📊 Vina Resource Statistics:")
        print(f"   Active instances: {stats['active_instances']}/{stats['max_instances']}")
        print(f"   Memory usage: {stats['current_memory_mb']:.1f}/{stats['memory_limit_mb']} MB ({stats['memory_percent']:.1f}%)")
        print(f"   Available memory: {stats['available_memory_mb']:.1f} MB")
        print(f"   Temp directories: {stats['temp_directories']}")
    else:
        print("📊 Vina resource manager not initialized")


# Register cleanup function for process exit
import atexit
atexit.register(cleanup_vina_resources)
