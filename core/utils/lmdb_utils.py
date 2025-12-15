"""
LMDB Utilities for Worker-Safe Database Access

This module provides utilities for safely accessing LMDB databases in multi-process
environments, particularly for PyTorch DataLoader workers.

The main issue this solves is the "DataLoader worker exited unexpectedly" error
that occurs when LMDB connections are shared across processes.
"""

import os
import lmdb
import threading
import pickle
from typing import Any, Optional, List


class WorkerSafeLMDBReader:
    """
    Worker-Safe LMDB Database Reader
    
    This class ensures that each worker process gets its own LMDB connection,
    preventing the "DataLoader worker exited unexpectedly" error that occurs
    when LMDB connections are shared across processes.
    
    Key Features:
    - Process/thread-aware connection management
    - Automatic reconnection on process/thread change
    - Cached length to avoid unnecessary DB access
    - Robust error handling for connection cleanup
    """
    
    def __init__(self, path: str, map_size: int = 10*(1024**3), max_readers: int = 256):
        """
        Initialize the LMDB reader
        
        Args:
            path: Path to the LMDB database
            map_size: Maximum size of the database (default: 10GB)
            max_readers: Maximum number of concurrent readers
        """
        self.path = path
        self.map_size = map_size
        self.max_readers = max_readers
        
        # Connection state
        self.db = None
        self.keys = None
        self._process_id = None
        self._thread_id = None
        self._keys_cache = None  # Cache keys to avoid repeated DB access for length
        
    def _get_process_thread_id(self):
        """Get current process and thread ID for connection management"""
        return (os.getpid(), threading.get_ident())
    
    def _should_reconnect(self):
        """Check if we need to reconnect due to process/thread change"""
        current_id = self._get_process_thread_id()
        return (self.db is None or 
                self._process_id != current_id[0] or 
                self._thread_id != current_id[1])

    def _connect_db(self):
        """
        Establish read-only database connection
        Safe for multi-process usage - each process gets its own connection
        """
        if not self._should_reconnect():
            return
            
        # Close existing connection if it exists
        if self.db is not None:
            try:
                self.db.close()
            except:
                pass  # Ignore errors when closing stale connections
            self.db = None
            self.keys = None
        
        # Establish new connection
        self.db = lmdb.open(
            self.path,
            map_size=self.map_size,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=self.max_readers,
        )
        
        # Update process/thread tracking
        self._process_id, self._thread_id = self._get_process_thread_id()
        
        # Load keys
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            
        # Cache keys for length operations
        if self._keys_cache is None:
            self._keys_cache = len(self.keys)

    def _close_db(self):
        """Close database connection"""
        if self.db is not None:
            try:
                self.db.close()
            except:
                pass  # Ignore errors when closing
            self.db = None
            self.keys = None
            self._process_id = None
            self._thread_id = None

    def __del__(self):
        """Cleanup on deletion"""
        self._close_db()

    def __len__(self):
        """
        Get dataset length without establishing full DB connection if possible
        Uses cached length to avoid triggering connection in parent process
        """
        # If we have cached length, use it
        if self._keys_cache is not None:
            return self._keys_cache
            
        # Otherwise, we need to connect to get the length
        self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx: int) -> Any:
        """
        Get item by index - ensures proper connection for current process/thread
        """
        # Ensure we have a valid connection for this process/thread
        self._connect_db()
        
        # Get the key and data
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        
        return data
    
    def get_keys(self) -> List[bytes]:
        """Get all keys in the database"""
        self._connect_db()
        return self.keys.copy()
    
    def get_raw(self, key: bytes) -> bytes:
        """Get raw data by key without unpickling"""
        self._connect_db()
        return self.db.begin().get(key)


def configure_multiprocessing_for_lmdb():
    """
    Configure multiprocessing settings for optimal LMDB compatibility
    
    This function should be called at the beginning of the main script
    to set up proper multiprocessing configuration.
    """
    import multiprocessing as mp
    import torch
    
    # Set multiprocessing start method to 'spawn' for better LMDB compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Set sharing strategy for better memory management
    torch.multiprocessing.set_sharing_strategy('file_system')


def get_safe_dataloader_config(num_workers: int, batch_size: int) -> dict:
    """
    Get safe DataLoader configuration for LMDB datasets
    
    Args:
        num_workers: Desired number of workers
        batch_size: Batch size
        
    Returns:
        Dictionary with safe DataLoader configuration
    """
    # Limit workers to prevent LMDB connection conflicts
    safe_num_workers = min(num_workers, 2) if num_workers > 0 else 0
    
    config = {
        'batch_size': batch_size,
        'num_workers': safe_num_workers,
        'pin_memory': True if safe_num_workers > 0 else False,
        'persistent_workers': True if safe_num_workers > 0 else False,
        'prefetch_factor': 2 if safe_num_workers > 0 else None,
        'multiprocessing_context': 'spawn' if safe_num_workers > 0 else None,
    }
    
    # Remove None values
    return {k: v for k, v in config.items() if v is not None}


class LMDBConnectionManager:
    """
    Context manager for LMDB connections
    
    Usage:
        with LMDBConnectionManager(path) as db:
            # Use db connection
            pass
    """
    
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.db = None
    
    def __enter__(self):
        self.db = lmdb.open(self.path, **self.kwargs)
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db is not None:
            try:
                self.db.close()
            except:
                pass
        self.db = None


def test_lmdb_connection(path: str) -> bool:
    """
    Test if LMDB database can be opened successfully
    
    Args:
        path: Path to LMDB database
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with LMDBConnectionManager(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        ) as db:
            with db.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
        return True
    except Exception as e:
        print(f"LMDB connection test failed for {path}: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("LMDB Utils - Worker-Safe Database Access")
    print("This module provides utilities for safely accessing LMDB databases")
    print("in multi-process environments like PyTorch DataLoader workers.")
