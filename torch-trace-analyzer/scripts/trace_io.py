#!/usr/bin/env python3

import gzip
import json
import os
import tarfile


def load_trace(filepath):
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(filepath, "r") as f:
            data = json.load(f)
    return data


def load_model_code(code_path):
    if code_path is None:
        return {}
    source_files = {}
    if code_path.endswith(".tar.gz") or code_path.endswith(".tgz"):
        with tarfile.open(code_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    content = tar.extractfile(member).read().decode("utf-8", errors="replace")
                    basename = os.path.basename(member.name)
                    source_files[basename] = content.split("\n")
    elif os.path.isdir(code_path):
        for root, _, files in os.walk(code_path):
            for f in files:
                if f.endswith(".py"):
                    fpath = os.path.join(root, f)
                    with open(fpath, "r", errors="replace") as fp:
                        source_files[f] = fp.read().split("\n")
    return source_files
