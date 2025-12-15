#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

# 3-letter -> 1-letter 映射，包含常见修饰残基的简化映射
AA_MAP = {
    # 标准20
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",

    # 修饰残基（按最接近的标准氨基酸处理）
    "FME":"M",  # N-formyl-Met
    "CME":"C",  # modified Cys
    "CSO":"C",  # Oxidized Cys
    "KCX":"K",  # Lys with carboxyl group
    # 其它不认识的残基可以按需求加
}

def pdb_to_seq(pdb_path: Path) -> str:
    """从 PDB 解析出一个简化的 AA 序列（所有链串在一起）."""
    residues = []
    seen = set()

    with open(pdb_path, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record != "ATOM":
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            icode  = line[26].strip()

            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            if resname in AA_MAP:
                residues.append(AA_MAP[resname])
            else:
                # 不认识的残基就跳过，也可以映射到 'X'
                # residues.append('X')
                continue

    return "".join(residues)


def main():
    root = Path("/data-extend/wangzilin/project/molpilot/MolPilot/data/test_set")
    fasta_path = Path("/data-extend/wangzilin/project/molpilot/MolPilot/data/crossdock_proteins.fasta")

    with open(fasta_path, "w") as fout:
        count = 0
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue

            # 例如 5S8I_2LY/5S8I_2LY_protein.pdb
            pdb_list = list(case_dir.glob("*_rec.pdb"))
            if not pdb_list:
                print(f"[WARN] no *_protein.pdb in {case_dir.name}")
                continue

            pdb_path = pdb_list[0]
            seq = pdb_to_seq(pdb_path)
            if len(seq) == 0:
                print(f"[WARN] empty seq for {pdb_path}")
                continue

            # 用 case_dir.name 作为 FASTA 的 ID，后面做 OOD 映射方便
            fasta_id = case_dir.name  # e.g. "5S8I_2LY"
            fout.write(f">{fasta_id}\n")
            fout.write(seq + "\n")
            count += 1

    print("Total PoseBusters protein sequences written:", count)
    print("FASTA saved to:", fasta_path)


if __name__ == "__main__":
    main()
