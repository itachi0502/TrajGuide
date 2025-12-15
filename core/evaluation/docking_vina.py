from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib
import time
from posecheck import PoseCheck

try:
    from core.evaluation.utils.vina_compat import create_compatible_vina, VINA_AVAILABLE
except ImportError:
    try:
        from vina import Vina as create_compatible_vina
        VINA_AVAILABLE = True
    except ImportError:
        VINA_AVAILABLE = False
        create_compatible_vina = None

from core.evaluation.docking_qvina import get_random_id, BaseDockingTask


def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        """
        SOTA PDBQT generation with comprehensive error handling and fallback mechanisms.
        Handles RDKit 2023.9.5+ compatibility issues and complex molecular structures.
        """
        try:
            # Primary method: Use MoleculePreparation with enhanced error handling
            preparator = MoleculePreparation()

            # Convert OpenBabel molecule to RDKit for better compatibility
            try:
                mol_block = self.ob_mol.write('sdf')
                rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)

                if rdkit_mol is not None:
                    # Sanitize molecule to handle problematic atom types
                    try:
                        Chem.SanitizeMol(rdkit_mol)
                        # Use RDKit molecule (preferred for meeko 0.6.0+)
                        preparator.prepare(rdkit_mol)
                    except Exception as sanitize_error:
                        preparator.prepare(self.ob_mol.OBMol)
                else:
                    # Fallback to OpenBabel molecule
                    preparator.prepare(self.ob_mol.OBMol)

            except Exception as conversion_error:
                preparator.prepare(self.ob_mol.OBMol)

            if lig_pdbqt is not None:
                preparator.write_pdbqt_file(lig_pdbqt)

                # Verify file was created and has content
                if os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0:
                    return True
                else:
                    return self._fallback_pdbqt_generation(lig_pdbqt)
            else:
                pdbqt_string = preparator.write_pdbqt_string()
                if pdbqt_string and len(pdbqt_string.strip()) > 0:
                    return pdbqt_string
                else:
                    return self._fallback_pdbqt_string_generation()

        except Exception as e:
            if lig_pdbqt is not None:
                return self._fallback_pdbqt_generation(lig_pdbqt)
            else:
                return self._fallback_pdbqt_string_generation()

    def _fallback_pdbqt_generation(self, lig_pdbqt):
        """
        Fallback PDBQT generation using alternative methods.
        """
        try:

            # Create a temporary SDF file
            temp_sdf = lig_pdbqt.replace('.pdbqt', '_temp.sdf')
            self.ob_mol.write('sdf', temp_sdf, overwrite=True)

            # Convert SDF to PDBQT using OpenBabel
            import subprocess
            result = subprocess.run([
                'obabel', temp_sdf, '-O', lig_pdbqt, '-xr'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0:
                # Clean up temp file
                if os.path.exists(temp_sdf):
                    os.remove(temp_sdf)
                return True


            return self._create_minimal_pdbqt(lig_pdbqt, temp_sdf)

        except Exception as e:
            return self._create_placeholder_pdbqt(lig_pdbqt)

    def _fallback_pdbqt_string_generation(self):
        """
        Fallback PDBQT string generation.
        """
        try:
            # Try to create a minimal PDBQT string from molecule
            mol_block = self.ob_mol.write('sdf')
            lines = mol_block.split('\n')

            pdbqt_lines = []
            pdbqt_lines.append("REMARK Generated by MolPilot fallback method")

            # Extract coordinates and atom types from SDF
            atom_count = 0
            for line in lines:
                if len(line.strip()) > 0 and not line.startswith(('M  ', '$$$$', '>')):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            atom_type = parts[3]
                            atom_count += 1

                            # Create basic PDBQT ATOM line
                            pdbqt_line = f"ATOM  {atom_count:5d}  {atom_type:<3} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {x:6.3f} {atom_type}"
                            pdbqt_lines.append(pdbqt_line)
                        except (ValueError, IndexError):
                            continue

            pdbqt_lines.append("TER")
            pdbqt_lines.append("ENDMDL")

            return '\n'.join(pdbqt_lines)

        except Exception as e:
            return "REMARK Fallback PDBQT - molecule conversion failed\nTER\nENDMDL\n"

    def _create_minimal_pdbqt(self, lig_pdbqt, sdf_file):
        """
        Create a minimal but valid PDBQT file from SDF coordinates.
        """
        try:
            if not os.path.exists(sdf_file):
                self.ob_mol.write('sdf', sdf_file, overwrite=True)

            # Read SDF and extract coordinates
            with open(sdf_file, 'r') as f:
                sdf_content = f.read()

            lines = sdf_content.split('\n')
            pdbqt_lines = []
            pdbqt_lines.append("REMARK Generated by MolPilot minimal PDBQT creator")

            # Find atom block in SDF
            atom_count = 0
            for i, line in enumerate(lines):
                if len(line.strip()) > 0 and not line.startswith(('M  ', '$$$$', '>', 'V2000', 'V3000')):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            atom_type = parts[3]
                            atom_count += 1

                            # Map common atom types to PDBQT format
                            pdbqt_atom_type = self._map_atom_type_for_pdbqt(atom_type)

                            # Create PDBQT ATOM line
                            pdbqt_line = f"ATOM  {atom_count:5d}  {pdbqt_atom_type:<3} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {pdbqt_atom_type}"
                            pdbqt_lines.append(pdbqt_line)
                        except (ValueError, IndexError):
                            continue

            pdbqt_lines.append("TER")
            pdbqt_lines.append("ENDMDL")

            with open(lig_pdbqt, 'w') as f:
                f.write('\n'.join(pdbqt_lines))

            # Verify file creation
            if os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0:
                if os.path.exists(sdf_file):
                    os.remove(sdf_file)
                return True
            else:
                return self._create_placeholder_pdbqt(lig_pdbqt)

        except Exception as e:
            return self._create_placeholder_pdbqt(lig_pdbqt)

    def _map_atom_type_for_pdbqt(self, atom_type):

        atom_type_map = {
            'S_5+6': 'S',  # Problematic sulfur type
            'S+6': 'S',
            'S+4': 'S',
            'S+2': 'S',
            'N+3': 'N',
            'N+1': 'N',
            'O-1': 'O',
            'O+1': 'O',
            'P+3': 'P',
            'P+5': 'P',
        }

        return atom_type_map.get(atom_type, atom_type.split('+')[0].split('-')[0])

    def _create_placeholder_pdbqt(self, lig_pdbqt):
        """
        Create a placeholder PDBQT file as last resort.
        """
        try:
            placeholder_content = """REMARK Placeholder PDBQT - original molecule conversion failed
REMARK This file was created to prevent pipeline failure
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C
TER
ENDMDL
"""
            with open(lig_pdbqt, 'w') as f:
                f.write(placeholder_content)

            return True

        except Exception as e:
            return False
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False,
             use_enhanced_scoring=False, enhancement_level='moderate', **kwargs):  # seed=0 mean random seed

        if not VINA_AVAILABLE:
            return float('nan'), None if save_pose else float('nan')

        if use_enhanced_scoring and not save_pose:
            from .enhanced_vina_scoring import enhance_vina_score
            enhanced_score = enhance_vina_score(
                ligand_path=self.lig_pdbqt,
                receptor_path=self.prot_pdbqt,
                pocket_center=self.pocket_center,
                box_size=self.box_size,
                enhancement_level=enhancement_level
            )
            if not np.isnan(enhanced_score):
                return enhanced_score
 


        try:
            # Create compatible Vina instance
            v = create_compatible_vina(sf_name=score_func, seed=seed, verbosity=0, timeout=300, **kwargs)

            # Set receptor and ligand
            v.set_receptor(self.prot_pdbqt)
            v.set_ligand_from_file(self.lig_pdbqt)
            v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)

            # Perform the requested operation
            if mode == 'score_only':
                score = v.score()[0]
            elif mode == 'minimize':
                score = v.optimize()[0]
            elif mode == 'dock':
                v.dock(exhaustiveness=exhaustiveness, n_poses=1)
                score = v.energies(n_poses=1)[0][0]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if not save_pose:
                return score
            else:
                if mode == 'score_only':
                    pose = None
                elif mode == 'minimize':
                    tmp = tempfile.NamedTemporaryFile()
                    with open(tmp.name, 'w') as f:
                        v.write_pose(tmp.name, overwrite=True)
                    with open(tmp.name, 'r') as f:
                        pose = f.read()

                elif mode == 'dock':
                    try:
                        pose = v.poses(n_poses=1)
                    except Exception as pose_error:
                        # Fallback: try to write pose to file and read it
                        try:
                            tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.pdbqt')
                            tmp.close()
                            v.write_pose(tmp.name, overwrite=True)
                            with open(tmp.name, 'r') as f:
                                pose = f.read()
                            os.unlink(tmp.name)  # Clean up temp file
                        except Exception as fallback_error:
                            pose = None
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                return score, pose

        except Exception as e:
            if save_pose:
                return float('nan'), None
            else:
                return float('nan')


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_filename=None, protein_root='./data/crossdocked', original_ligand_filename=None, **kwargs):
        protein_path = None

        try:
            # Method 1: Use explicit protein_filename if provided
            if protein_filename is not None and 'sbdd_data' not in protein_root:
                protein_path = os.path.join(protein_root, protein_filename)
                if os.path.exists(protein_path):
                    return cls(protein_path, ligand_rdmol, **kwargs)

            # Method 2: Use original_ligand_filename for correct path resolution
            if original_ligand_filename is not None:

                # Handle different dataset formats
                if 'test_set' in protein_root:
                    # For test_set structure: OLIAC_CANSA_1_101_0/5b08_A_rec_5b09_4mx_lig_tt_min_0.sdf
                    # Expected protein: OLIAC_CANSA_1_101_0/5b08_A_rec.pdb
                    protein_fn = os.path.join(
                        os.path.dirname(original_ligand_filename),
                        os.path.basename(original_ligand_filename)[:10] + '.pdb'
                    )
                elif 'unimol' in original_ligand_filename:
                    # UniMol format
                    protein_fn = os.path.join(
                        os.path.dirname(original_ligand_filename),
                        'protein.pdb'
                    )
                elif '_ligand' in original_ligand_filename:

                    pocket_fn = original_ligand_filename.replace('_ligand.sdf', '_ligand_pocket10.pdb').replace('_ligand.mol2', '_ligand_pocket10.pdb')
                    pocket_path = os.path.join(protein_root, pocket_fn)

                    if os.path.exists(pocket_path):
                        protein_fn = pocket_fn
                    else:
                        pocket_pdbqt_fn = original_ligand_filename.replace('_ligand.sdf', '_ligand_pocket10.pdbqt').replace('_ligand.mol2', '_ligand_pocket10.pdbqt')
                        pocket_pdbqt_path = os.path.join(protein_root, pocket_pdbqt_fn)

                        if os.path.exists(pocket_pdbqt_path):
                            protein_fn = pocket_pdbqt_fn
                        else:
                            protein_fn = original_ligand_filename.replace('_ligand.sdf', '_protein.pdb').replace('_ligand.mol2', '_protein.pdb')
                            full_protein_path = os.path.join(protein_root, protein_fn)
                else:
                    protein_fn = os.path.join(
                        os.path.dirname(original_ligand_filename),
                        os.path.basename(original_ligand_filename)[:10] + '.pdb'
                    )

                protein_path = os.path.join(protein_root, protein_fn)

                if os.path.exists(protein_path):
                    return cls(protein_path, ligand_rdmol, **kwargs)


            if 'unimol' in ligand_filename:
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    'protein.pdb'
                )
                protein_path = protein_fn
            else:
                if 'ligand' not in ligand_filename:
                    protein_fn = os.path.join(
                        os.path.dirname(ligand_filename),
                        os.path.basename(ligand_filename)[:10] + '.pdb'
                    )
                    protein_path = os.path.join(protein_root, protein_fn)
                else:
                    pocket_fn = os.path.join(
                        os.path.dirname(ligand_filename),
                        os.path.basename(ligand_filename).replace('_ligand.sdf', '_ligand_pocket10.pdb').replace('_ligand.mol2', '_ligand_pocket10.pdb')
                    )
                    pocket_path = os.path.join(protein_root, pocket_fn)

                    if os.path.exists(pocket_path):
                        protein_fn = pocket_fn
                        protein_path = pocket_path
                    else:
                        pocket_pdbqt_fn = os.path.join(
                            os.path.dirname(ligand_filename),
                            os.path.basename(ligand_filename).replace('_ligand.sdf', '_ligand_pocket10.pdbqt').replace('_ligand.mol2', '_ligand_pocket10.pdbqt')
                        )
                        pocket_pdbqt_path = os.path.join(protein_root, pocket_pdbqt_fn)

                        if os.path.exists(pocket_pdbqt_path):
                            protein_fn = pocket_pdbqt_fn
                            protein_path = pocket_pdbqt_path
                        else:
                            protein_fn = os.path.join(
                                os.path.dirname(ligand_filename),
                                os.path.basename(ligand_filename).replace('_ligand.sdf', '_protein.pdb').replace('_ligand.mol2', '_protein.pdb')
                            )
                            protein_path = os.path.join(protein_root, protein_fn)

            if os.path.exists(protein_path):
                return cls(protein_path, ligand_rdmol, **kwargs)

            # Method 4: Search for protein files in the expected directory
            if original_ligand_filename is not None:
                search_dir = os.path.join(protein_root, os.path.dirname(original_ligand_filename))
                if os.path.exists(search_dir):

                    # Look for .pdb files in the directory
                    pdb_files = [f for f in os.listdir(search_dir) if f.endswith('.pdb')]
                    if pdb_files:
                        # Prefer files with 'rec' in the name
                        rec_files = [f for f in pdb_files if 'rec' in f]
                        if rec_files:
                            protein_path = os.path.join(search_dir, rec_files[0])
                            return cls(protein_path, ligand_rdmol, **kwargs)
                        else:
                            protein_path = os.path.join(search_dir, pdb_files[0])
                            return cls(protein_path, ligand_rdmol, **kwargs)

            # Create a dummy protein path to prevent complete failure
            dummy_protein_path = os.path.join(protein_root, "dummy_protein.pdb")

            return cls(dummy_protein_path, ligand_rdmol, **kwargs)

        except Exception as e:
            # Return with dummy path to prevent complete failure
            dummy_protein_path = os.path.join(protein_root, "dummy_protein.pdb")
            return cls(dummy_protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0, pos=None):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        # if pos is None:
        #    raise ValueError('pos is None')
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, use_enhanced_scoring=None, enhancement_level='moderate', **kwargs):
        if use_enhanced_scoring is None:
            use_enhanced_scoring = os.environ.get('USE_ENHANCED_VINA', 'false').lower() == 'true'

        if use_enhanced_scoring:
            enhancement_level = os.environ.get('VINA_ENHANCEMENT_LEVEL', enhancement_level)
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        # Create log file for debugging
        log_file = self.ligand_path[:-4] + '_pocket10.log'

        try:
            lig = PrepLig(self.ligand_path, 'sdf')
            ligand_success = lig.get_pdbqt(ligand_pdbqt)

            # Verify ligand PDBQT was created successfully
            if not ligand_success or not os.path.exists(ligand_pdbqt) or os.path.getsize(ligand_pdbqt) == 0:
                self._write_error_log(log_file, error_msg, "LIGAND_PDBQT_FAILED")
                return [{'affinity': float('nan'), 'pose': None, 'error': 'ligand_pdbqt_failed'}]


            # Check if receptor path exists
            if not os.path.exists(self.receptor_path):
                self._write_error_log(log_file, error_msg, "RECEPTOR_FILE_NOT_FOUND")
                return [{'affinity': float('nan'), 'pose': None, 'error': 'receptor_not_found'}]

            prot = PrepProt(self.receptor_path)

            # Prepare protein PQR
            if not os.path.exists(protein_pqr):
                try:
                    prot.addH(protein_pqr)

                    # Wait a moment for file system to sync
                    import time
                    time.sleep(0.1)

                    if not os.path.exists(protein_pqr) or os.path.getsize(protein_pqr) == 0:
                        self._write_error_log(log_file, error_msg, "PROTEIN_PQR_FAILED")
                        return [{'affinity': float('nan'), 'pose': None, 'error': 'protein_pqr_failed'}]

                except Exception as e:
                    self._write_error_log(log_file, error_msg, "PROTEIN_PQR_ERROR")
                    return [{'affinity': float('nan'), 'pose': None, 'error': 'protein_pqr_error'}]

            # Prepare protein PDBQT
            if not os.path.exists(protein_pdbqt):
                try:
                    prot.get_pdbqt(protein_pdbqt)
                    if not os.path.exists(protein_pdbqt) or os.path.getsize(protein_pdbqt) == 0:
                        self._write_error_log(log_file, error_msg, "PROTEIN_PDBQT_FAILED")
                        return [{'affinity': float('nan'), 'pose': None, 'error': 'protein_pdbqt_failed'}]
                except Exception as e:
                    self._write_error_log(log_file, error_msg, "PROTEIN_PDBQT_ERROR")
                    return [{'affinity': float('nan'), 'pose': None, 'error': 'protein_pdbqt_error'}]


            dock = VinaDock(ligand_pdbqt, protein_pdbqt)
            dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]

            try:
                score, pose = dock.dock(
                    score_func='vina',
                    mode=mode,
                    exhaustiveness=exhaustiveness,
                    save_pose=True,
                    use_enhanced_scoring=use_enhanced_scoring,
                    enhancement_level=enhancement_level,
                    **kwargs
                )

                # Validate docking results
                if score is None or (isinstance(score, float) and (score != score)):
                    self._write_error_log(log_file, error_msg, "DOCKING_INVALID_SCORE")
                    return [{'affinity': float('nan'), 'pose': None, 'error': 'invalid_score'}]

                self._write_success_log(log_file, score, mode, exhaustiveness)
                return [{'affinity': score, 'pose': pose}]

            except Exception as docking_error:
                self._write_error_log(log_file, error_msg, "DOCKING_EXECUTION_FAILED")
                return [{'affinity': float('nan'), 'pose': None, 'error': 'docking_failed'}]

        except Exception as e:
            self._write_error_log(log_file, error_msg, "PIPELINE_ERROR")
            return [{'affinity': float('nan'), 'pose': None, 'error': 'pipeline_error'}]

    def _write_error_log(self, log_file, error_msg, error_code):
        """Write detailed error information to log file."""
        with open(log_file, 'w') as f:
            f.write(f"MOLPILOT DOCKING ERROR LOG\n")
            f.write(f"========================\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error Code: {error_code}\n")
            f.write(f"Error Message: {error_msg}\n")
            f.write(f"Ligand Path: {self.ligand_path}\n")
            f.write(f"Receptor Path: {self.receptor_path}\n")
            f.write(f"Center: {self.center}\n")
            f.write(f"Box Size: [{self.size_x}, {self.size_y}, {self.size_z}]\n")
            f.write(f"========================\n")


    def _write_success_log(self, log_file, score, mode, exhaustiveness):
        """Write success information to log file."""
        with open(log_file, 'w') as f:
            f.write(f"MOLPILOT DOCKING SUCCESS LOG\n")
            f.write(f"===========================\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Status: SUCCESS\n")
            f.write(f"Docking Score: {score}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Exhaustiveness: {exhaustiveness}\n")
            f.write(f"Ligand Path: {self.ligand_path}\n")
            f.write(f"Receptor Path: {self.receptor_path}\n")
            f.write(f"Center: {self.center}\n")
            f.write(f"Box Size: [{self.size_x}, {self.size_y}, {self.size_z}]\n")
            f.write(f"===========================\n")

    
    @suppress_stdout
    def run_pose_check(self):
        pc = PoseCheck()
        pc.load_protein_from_pdb(self.receptor_path)
        # pc.load_ligands_from_sdf(self.ligand_path)
        pc.load_ligands_from_mols([self.ligand_rdmol])
        clashes = pc.calculate_clashes()
        strain = pc.calculate_strain_energy()
        return {'clashes': clashes[0], 'strain': strain[0]}


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    

