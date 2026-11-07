"""Build adversarial show folders from the SYNTHETIC fixtures only (never the confidential pack)."""
import re, shutil, sys, zipfile
from pathlib import Path

FIX = Path(sys.argv[1]); OUT = Path(sys.argv[2])
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
e1 = (FIX / "episode-001-synthetic.txt").read_text(); e2 = (FIX / "episode-002-synthetic.txt").read_text()

def show(name, files):
    d = OUT / name; d.mkdir()
    for fn, txt in files.items():
        p = d / fn
        if isinstance(txt, bytes): p.write_bytes(txt)
        else: p.write_text(txt, encoding="utf-8")
    return d

def renum(txt, n):
    return re.sub(r"EPISODE NUMBER: Episode \d+", f"EPISODE NUMBER: Episode {n}", txt)

show("baseline", {"ep1.txt": e1, "ep2.txt": e2})
show("crlf_bom", {"ep1.txt": ("﻿" + e1).replace("\n", "\r\n"), "ep2.txt": e2.replace("\n", "\r\n")})
hindi = re.sub(r'"I am marrying your sister."', '"मैं तुम्हारी बहन से शादी कर रहा हूँ।"', e1)
hindi = re.sub(r"\*\*MARA \(Hopeful\):\*\*", "**मीरा (उम्मीद से):**", hindi)
show("hindi", {"ep1.txt": hindi, "ep2.txt": e2})
nocam = re.sub(r"\*\*CAMERA:\*\*\n[^\n]*\n[^\n]*\n\n?", "", e1)
show("no_camera", {"ep1.txt": nocam, "ep2.txt": re.sub(r"\*\*CAMERA:\*\*\n[^\n]*\n[^\n]*\n\n?", "", e2)})
hh = re.sub(r"\((\d\d):(\d\d\.\d\d)\)", r"(00:\1:\2)", e1)
show("hhmmss", {"ep1.txt": hh, "ep2.txt": e2})
show("comma_decimals", {"ep1.txt": re.sub(r"\((\d\d):(\d\d)\.(\d\d)\)", r"(\1:\2,\3)", e1), "ep2.txt": e2})
show("single_episode", {"ep1.txt": e1})
show("one_file_all", {"all_episodes.txt": e1 + "\n\n" + e2})
show("no_header", {"episode-007.txt": re.sub(r"EPISODE NUMBER:[^\n]*\n", "", e1), "episode-008.txt": re.sub(r"EPISODE NUMBER:[^\n]*\n", "", e2)})
show("gap_numbers", {"ep1.txt": e1, "ep3.txt": renum(e2, 3)})
show("numbers_101", {"ep1.txt": renum(e1, 101), "ep2.txt": renum(e2, 102)})
weird = e1.replace("**MUSIC:**", "**NARRATOR (V.O.):**").replace("**ACTION:**", "**TEXT ON SCREEN:**", 1)
show("weird_labels", {"ep1.txt": weird, "ep2.txt": e2})
show("cp1252", {"ep1.txt": e1.replace('"Tom?"', "“Tom?” — café").encode("cp1252"), "ep2.txt": e2})
show("empty_file", {"ep1.txt": e1, "ep2.txt": e2, "notes.txt": ""})
show("md_files", {"ep1.md": e1, "ep2.md": e2})
# a zip with macOS junk and a nested folder
z = OUT / "zipped.zip"
with zipfile.ZipFile(z, "w") as zf:
    zf.writestr("show/ep1.txt", e1); zf.writestr("show/ep2.txt", e2)
    zf.writestr("__MACOSX/show/._ep1.txt", "junk"); zf.writestr("show/.DS_Store", "junk")
# no dialogue at all
nodia = "\n".join(l for l in e1.split("\n\n") if not re.match(r"\*\*[A-Z]+ \(", l.strip()))
show("no_dialogue", {"ep1.txt": "\n\n".join(b for b in e1.split("\n\n") if not re.match(r"\*\*(MARA|TOM|LENA)", b.strip())), "ep2.txt": e2})
# shorter than 20 s total
short = re.sub(r"\((\d\d):(\d\d\.\d\d)\)", lambda m: "(00:%05.2f)" % (float(m.group(2)) / 10 + int(m.group(1)) * 6), e1)
show("too_short", {"ep1.txt": short})
show("garbage", {"ep1.txt": "this is not a screenplay\njust some notes\n" * 30})
print("ok", sorted(p.name for p in OUT.iterdir()))
