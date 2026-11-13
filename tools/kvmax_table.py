import json, os, re, subprocess
rows=[]
for d in sorted(os.listdir("ops")):
    if not (d.startswith("kvmax_decode") or d.startswith("kvmax_1l")): continue
    p=os.path.join("ops",d)
    mp=os.path.join(p,"model.json")
    if not os.path.exists(mp): continue
    m=json.load(open(mp))
    C=m["shape"]["C"]; L=m["layers"]; T=m["shape"]["T"]
    log=os.path.join(p,"model.compile.log")
    ex=step=dram=None; size=None
    if os.path.exists(log):
        t=open(log,errors="replace").read()
        mm=re.search(r"EXIT=(\d+)",t); ex=mm.group(1) if mm else None
        ms=re.search(r"SIZE=(\d+)",t); size=ms.group(1) if ms else None
        mst=re.search(r"has (\d+) Execution Step",t); step=mst.group(1) if mst else None
        md=re.search(r"\[ 0\]: MDLA 5\.0\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",t)
        dram=" / ".join(md.groups()) if md else None
        if re.search(r"Cannot find an execution plan",t): ex=ex or "1"; step="(backend) unsupported ops"
    kv=2*int(m["shape"]["KLayers"]) if "KLayers" in m["shape"] else None
    rows.append((d,L,T,C,m.get("tflite_bytes"),size,ex,step,dram))
print("%-30s %2s %2s %7s %11s %11s %5s %4s %s"%("dir","L","T","C","tflite","dla","EXIT","st","DRAM Input/Output/Temp/Static/Code/Total"))
for r in rows:
    print("%-30s %2d %2d %7d %11s %11s %5s %4s %s"%(r[0],r[1],r[2],r[3],r[4],r[5] or "-",r[6] or "-",r[7] or "-",r[8] or "-"))
print()
print("KV bytes (K+V): C -> 2*6144*C")
for r in rows:
    print("  C=%6d  %d B (%.1f MiB)"%(r[3], 12288*r[3], 12288*r[3]/1048576))
