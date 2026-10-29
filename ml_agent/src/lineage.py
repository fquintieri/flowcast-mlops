import hashlib
import os

import pandas as pd

# Repositorio montado read-only pelo Airflow/compose; o binario git nao existe na imagem.
GIT_DIR = "/app/.git"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def get_git_commit(git_dir=GIT_DIR):
    """Hash do commit em HEAD lido direto do .git, ou None se indisponivel.

    Nao detecta alteracoes nao commitadas: o codigo roda a partir de um bind mount.
    """
    head_path = os.path.join(git_dir, "HEAD")
    if not os.path.isfile(head_path):
        return None

    head = _read(head_path)
    if not head.startswith("ref: "):
        return head  # HEAD destacado: ja e o hash

    ref = head[len("ref: "):]
    ref_path = os.path.join(git_dir, ref)
    if os.path.isfile(ref_path):
        return _read(ref_path)

    packed_refs = os.path.join(git_dir, "packed-refs")
    if os.path.isfile(packed_refs):
        with open(packed_refs, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return None


def fingerprint_dataframe(df, sort_by):
    """SHA-256 deterministico do conteudo (independe da ordem de leitura das linhas)."""
    ordered = df.sort_values(sort_by).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()
