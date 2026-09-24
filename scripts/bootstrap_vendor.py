"""Fetch unchanged third-party source from the recorded upstream commit."""
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]/'ipad/vendor'
manifest=json.loads((root/'PROVENANCE.json').read_text())
for name,digest in manifest['files'].items():
    path=root/name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
    url=f"https://raw.githubusercontent.com/LJF1113/IPAD/{manifest['commit']}/model/{name}"
    data=urllib.request.urlopen(url).read()
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError(f'Upstream hash mismatch: {name}')
    path.write_bytes(data)
    print(f'Verified {name}: {digest}')
