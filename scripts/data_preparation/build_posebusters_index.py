import os, pickle
from pathlib import Path

root = Path("/data-extend/wangzilin/project/molpilot/MolPilot/data/posebusters_benchmark_set")  # 你的 source 目录
index = []

for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue

    cid = case_dir.name  # 比如 "5S8I_2LY"

    # 1) 找蛋白：*_protein.pdb
    protein_list = list(case_dir.glob("*_protein.pdb"))
    if not protein_list:
        print(f"[WARN] skip {cid}: no *_protein.pdb")
        continue
    protein_rel = protein_list[0].relative_to(root)

    # 2) 找主配体：*_ligand.sdf，但排除 *_ligands.sdf 和 *_ligand_start_conf.sdf
    ligand_candidates = list(case_dir.glob("*_ligand.sdf"))
    ligand_candidates = [
        p for p in ligand_candidates
        if not p.name.endswith("_ligands.sdf")
        and not p.name.endswith("_ligand_start_conf.sdf")
    ]
    if not ligand_candidates:
        print(f"[WARN] skip {cid}: no main *_ligand.sdf")
        continue
    ligand_rel = ligand_candidates[0].relative_to(root)

    # 3) rmsd 对 PoseBusters 不重要，设成 0.0 即可
    index.append((str(protein_rel), str(ligand_rel), 0.0))

print("Total complexes:", len(index))
with open(root / "index.pkl", "wb") as f:
    pickle.dump(index, f)
print("Saved to", root / "index.pkl")


