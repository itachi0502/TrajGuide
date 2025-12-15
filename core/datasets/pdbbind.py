import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import threading

from core.datasets.utils import PDBProtein
from core.datasets.protein_ligand import parse_sdf_file_mol
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from scipy import stats


class PDBBindDataset(Dataset):
    """
    Worker-Safe PDBBind Dataset with LMDB backend

    This class ensures that each worker process gets its own LMDB connection,
    preventing the "DataLoader worker exited unexpectedly" error.
    """

    def __init__(self, raw_path, transform=None, emb_path=None, heavy_only=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(self.raw_path, os.path.basename(self.raw_path) + '_processed.lmdb')
        self.emb_path = emb_path
        self.transform = transform
        self.heavy_only = heavy_only
        self.db = None
        self.keys = None

        # Worker-safe connection management
        self._process_id = None
        self._thread_id = None
        self._keys_cache = None

        if not os.path.exists(self.processed_path):
            self._process()
        print('Load dataset from ', self.processed_path)
        if self.emb_path is not None:
            print('Load embedding from ', self.emb_path)
            self.emb = torch.load(self.emb_path)

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
            self.processed_path,
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
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        # index = parse_pdbbind_index_file(self.index_path)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, resolution, pka, kind) in enumerate(tqdm(index)):
                # try:
                # pdb_path = os.path.join(self.raw_path, 'refined-set', pdb_idx)
                # pocket_fn = os.path.join(pdb_path, f'{pdb_idx}_pocket.pdb')
                # ligand_fn = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')
                pocket_dict = PDBProtein(pocket_fn).to_dict_atom()
                ligand_dict = parse_sdf_file_mol(ligand_fn, heavy_only=self.heavy_only)
                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )
                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                data.y = torch.tensor(float(pka))
                data.kind = torch.tensor(kind)
                txn.put(
                    key=f'{i:05d}'.encode(),
                    value=pickle.dumps(data)
                )
                # except:
                #     num_skipped += 1
                #     print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                #     continue
        print('num_skipped: ', num_skipped)
    
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
        data.id = idx
        assert data.protein_pos.size(0) > 0

        if self.transform is not None:
            data = self.transform(data)

        # add features extracted by molopt
        if self.emb_path is not None:
            emb = self.emb[idx]
            data.nll = torch.cat([emb['kl_pos'][1:], emb['kl_v'][1:]]).view(1, -1)
            data.nll_all = torch.cat([emb['kl_pos'], emb['kl_v']]).view(1, -1)
            data.pred_ligand_v = torch.softmax(emb['pred_ligand_v'], dim=-1)
            data.final_h = emb['final_h']
            # data.final_ligand_h = emb['final_ligand_h']
            data.pred_v_entropy = torch.from_numpy(
                stats.entropy(torch.softmax(emb['pred_ligand_v'], dim=-1).numpy(), axis=-1)).view(-1, 1)

        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PDBBindDataset(args.path)
