"""Compile every experiment record into one dashboard: DASHBOARD.md.

Sources: zoo/EXPERIMENTS*.md tables, LEVERS.md statuses, logs/run_*/summary.json.
Regenerate any time: uv run python tools/ledger.py
"""
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    out = ["# Dashboard — everything tried, generated. Do not edit.\n"]
    # experiment tallies
    total = wins = 0
    out.append("## Measured experiment cells\n")
    out.append("| campaign file | cells | confirmed wins | last update |")
    out.append("|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(ROOT, "zoo", "EXPERIMENTS*.md"))):
        text = open(f).read()
        rows = [l for l in text.splitlines() if l.startswith("|") and re.search(r"\d\.\d{3}", l)]
        cells = len(rows)
        w = len(re.findall(r"confirmed win|ACCEPTED|\*\*confirmed", text))
        total += cells; wins += w
        import datetime
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%d %b %H:%M")
        out.append(f"| {os.path.basename(f)} | {cells} | {w} | {mt} |")
    out.append(f"\n**~{total} table rows logged, {wins} confirmed-win markers.**\n")
    # autonomous runs
    out.append("## Autonomous runs\n")
    out.append("| run | stop | iters | best primary | wall min | tokens |")
    out.append("|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(ROOT, "logs", "run_*", "summary.json"))):
        s = json.load(open(f))
        out.append(f"| {os.path.basename(os.path.dirname(f))} | {s.get('stop_reason')} "
                   f"| {s.get('iterations')} | {round(s['best_metrics']['primary'],4)} "
                   f"| {round(s.get('wall_s',0)/60,1)} | {s.get('tokens_total','?')} |")
    # levers by status
    out.append("\n## Lever status (from LEVERS.md)\n")
    text = open(os.path.join(ROOT, "LEVERS.md")).read()
    buckets = {"DEAD": [], "ALIVE": [], "IN-FLIGHT": [], "PARKED": [], "N/A": [], "TODO/QUEUED": []}
    for line in text.splitlines():
        if not line.startswith("- "): continue
        name = line[2:].split("—")[0].strip()
        if re.search(r"\bDEAD\b", line): buckets["DEAD"].append(name)
        elif re.search(r"\bALIVE\b", line): buckets["ALIVE"].append(name)
        elif re.search(r"\bC[1-9]\b|running|building|QUEUED behind", line): buckets["IN-FLIGHT"].append(name)
        elif "PARKED" in line: buckets["PARKED"].append(name)
        elif "N/A" in line: buckets["N/A"].append(name)
        elif re.search(r"TODO|QUEUED", line): buckets["TODO/QUEUED"].append(name)
    for k, v in buckets.items():
        out.append(f"### {k} ({len(v)})")
        out.extend(f"- {x}" for x in v)
        out.append("")
    out.append("## Anything untried?\n")
    open_items = buckets["IN-FLIGHT"] + buckets["TODO/QUEUED"]
    out.append("Open items above are the complete untried set. If an idea is not in "
               "LEVERS.md at all, it is UNRECORDED — add it there first, that is the rule.")
    with open(os.path.join(ROOT, "DASHBOARD.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"DASHBOARD.md written: {total} cells, {len(buckets['IN-FLIGHT'])} in-flight, "
          f"{len(buckets['DEAD'])} dead, {len(buckets['ALIVE'])} alive")

if __name__ == "__main__":
    main()
