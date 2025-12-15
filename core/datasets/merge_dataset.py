import os
import pickle
import lmdb
from tqdm.auto import tqdm
from core.datasets.pl_data import ProteinLigandData
import torch
import threading

class DBReader:
    """
    Worker-Safe LMDB Database Reader for Merge Dataset

    This class ensures that each worker process gets its own LMDB connection,
    preventing the "DataLoader worker exited unexpectedly" error that occurs
    when LMDB connections are shared across processes.
    """
    def __init__(self, path) -> None:
        self.path = path
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
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Update process/thread tracking
        self._process_id, self._thread_id = self._get_process_thread_id()

        # Load keys
        with self.db.begin() as txn:
            self.keys = sorted(list(txn.cursor().iternext(values=False)))

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

    def __getitem__(self, idx):
        """
        Get item by index - ensures proper connection for current process/thread
        """
        # Ensure we have a valid connection for this process/thread
        self._connect_db()

        # Get the key and data
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        # Convert to ProteinLigandData
        data = ProteinLigandData(**data)
        data.id = idx
        if hasattr(data, 'protein_pos'):
            assert data.protein_pos.size(0) > 0, f'Empty protein_pos: {data.ligand_filename}, {data.protein_pos.size()}'
        return data

    def add(self, add_db):
        if self.db is None:
            self._connect_db()
        offset = len(self)
        with self.db.begin(write=True, buffers=True) as txn:
            for i in tqdm(range(len(add_db)), desc='Merging'):
                data = add_db[i]
                data.id = i + offset
                txn.put(str(data.id).encode(), pickle.dumps(data))
        self._close_db()
        

if __name__ == '__main__':
    dst_dir = './data/'
    dst_path = os.path.join(dst_dir, 'PDB+MOAD_processed_final.lmdb')
    pdb_path = os.path.join(dst_dir, 'PDBBind_processed_final.lmdb')
    moad_path = os.path.join(dst_dir, 'BindingMOAD_2020_pocket10_processed_final.lmdb')
    dataset_pdb = DBReader(pdb_path)
    dataset_moad = DBReader(moad_path)

    train_names = torch.load('./data/PDB+MOAD_train_names.pt')
    test_names = torch.load('./data/PDB+MOAD_test_names.pt') 

    pose_split = {'train': [], 'test': []}
    num_skipped = 0

    # db = lmdb.open(dst_path, map_size=10*(1024*1024*1024), create=True, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=256)
    # with db.begin(write=True, buffers=True) as txn:
    #     idx = -1
    #     for i in tqdm(range(len(dataset_pdb)), desc='Merging PDB'):
    #         idx += 1
    #         try:
    #             data = dataset_pdb[i]
    #             data.id = idx
    #             data = data.to_dict()
    #             txn.put(str(idx).encode(), pickle.dumps(data))
    #             protein_fn = data.protein_filename
    #             ligand_fn = data.ligand_filename
    #             if (protein_fn, ligand_fn) in train_names:
    #                 pose_split['train'].append(idx)
    #             elif (protein_fn, ligand_fn) in test_names:
    #                 pose_split['test'].append(idx)
    #         except Exception as e:
    #             num_skipped += 1
    #             # print(f'Error: {e}, {i}')
    #     for i in tqdm(range(len(dataset_moad)), desc='Merging MOAD'):
    #         idx += 1
    #         try:
    #             data = dataset_moad[i]
    #             data.id = idx
    #             data = data.to_dict()
    #             txn.put(str(idx).encode(), pickle.dumps(data))
    #             protein_fn = data.protein_filename
    #             ligand_fn = data.ligand_filename
    #             if (protein_fn, ligand_fn) in train_names:
    #                 pose_split['train'].append(idx)
    #             elif (protein_fn, ligand_fn) in test_names:
    #                 pose_split['test'].append(idx)
    #         except Exception as e:
    #             num_skipped += 1
    #             # print(f'Error: {e}, {i}')
        
    # db.close()

    ############################################################

    dataset = DBReader(dst_path)
    print(len(dataset), dataset[0])

    pose_split_prev = torch.load('./data/PDB+MOAD_pose_split_filtered_qed17.pt')
    
    for i in tqdm(pose_split_prev['train'], desc='Checking Train'):
        try:
            data = dataset[i]
            protein_fn = data.protein_filename
            ligand_fn = data.ligand_filename
            charge = data.ligand_charge.tolist()
            # if any charge is not 0 / -1 / 1, skip
            if any([c not in [-1, 0, 1] for c in charge]):
                num_skipped += 1
                continue
            
            assert (protein_fn, ligand_fn) in train_names, f'Not in train: {i}, {protein_fn}, {ligand_fn}'
            pose_split['train'].append(i)
        except Exception as e:
            num_skipped += 1
            # print(f'Error: {e}, {i}')

    for i in tqdm(pose_split_prev['test'], desc='Checking Test'):
        try:
            data = dataset[i]
            protein_fn = data.protein_filename
            ligand_fn = data.ligand_filename
            charge = data.ligand_charge.tolist()
            # if any charge is not 0 / -1 / 1, skip
            if any([c not in [-1, 0, 1] for c in charge]):
                num_skipped += 1
                continue
            
            assert (protein_fn, ligand_fn) in test_names, f'Not in test: {i}, {protein_fn}, {ligand_fn}'
            pose_split['test'].append(i)
        except Exception as e:
            num_skipped += 1
            # print(f'Error: {e}, {i}')

    print(len(train_names), len(test_names))
    torch.save(pose_split, './data/PDB+MOAD_pose_split_filtered_qed17_charge.pt')
    print(f'Train: {len(pose_split["train"])}, Test: {len(pose_split["test"])}, Skipped: {num_skipped}')