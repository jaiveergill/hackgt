#!/usr/bin/env python3
"""Summarize a session log (logs/latest.jsonl by default): camera fps and gaps, link ping, stalls, decode latency, voice, errors.

  python scripts/session_report.py               # latest session
  python scripts/session_report.py logs/session_20260926_213000.jsonl
  python scripts/session_report.py --events       # also dump the timeline of notable events
"""
import json, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(len(xs) * q), len(xs) - 1)] if xs else None


def fmt_ms(x):
    return "-" if x is None else f"{x*1000:.0f} ms"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "logs", "latest.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("empty log"); return
    by = {}
    for r in rows:
        by.setdefault(r["ev"], []).append(r)
    start = by.get("start", [{}])[0]
    dur = rows[-1]["rel"]
    print(f"session {os.path.basename(os.path.realpath(path))}  ·  {dur/60:.1f} min  ·  git {start.get('git')}  ·  source {start.get('args', {}).get('source')}  ·  voice {start.get('args', {}).get('voice')}")
    co = by.get("camera_opened", [])
    if co:
        print(f"camera: opened={co[0].get('opened')} {co[0].get('source')}")

    # ---- camera frames
    cs = by.get("camera_stats", [])
    if cs:
        fps = [c["fps"] for c in cs]; p95 = [c["p95_ms"] for c in cs]; mx = [c["max_ms"] for c in cs]
        gaps = [g for c in cs for g in c.get("gaps", [])]
        n_gaps = sum(c.get("n_gaps", 0) for c in cs)
        face = [c.get("face_rate", 0) for c in cs]
        print(f"\nCAMERA ({len(cs)} x 5 s windows)")
        print(f"  fps        median {st.median(fps):.1f}   min {min(fps):.1f}   (windows under 12 fps: {sum(f < 12 for f in fps)})")
        print(f"  interval   p95 median {st.median(p95):.0f} ms   worst frame gap {max(mx)/1000:.2f} s")
        print(f"  gaps>0.3s  {n_gaps} total = {n_gaps/(len(cs)*5/60):.1f} per minute;  longest: {sorted(gaps, reverse=True)[:8]}")
        print(f"  face seen  {st.median(face)*100:.0f}% of frames (median window)")
        bad = [c for c in cs if c["max_ms"] > 1000]
        if bad:
            print("  windows with a gap over 1 s at (min:sec):", ", ".join(f"{int(c['rel']//60)}:{int(c['rel']%60):02d}({c['max_ms']/1000:.1f}s)" for c in bad[:12]))
    else:
        print("\nCAMERA: no frame stats (capture process never reported)")

    # ---- link
    pg = by.get("ping", [])
    if pg:
        rt = [r for p in pg for r in p.get("rtt_ms", [])]
        loss = [p.get("loss_pct") or 0 for p in pg]
        print(f"\nLINK to {pg[0].get('host')} ({len(pg)} probes)")
        if rt:
            print(f"  ping       avg {st.mean(rt):.0f} ms   p95 {pct(rt, .95):.0f} ms   max {max(rt):.0f} ms   spikes>150ms {sum(r > 150 for r in rt)}/{len(rt)}   probes with loss {sum(l > 0 for l in loss)}")
        else:
            print("  ping: no replies at all")
    ce = by.get("camera_error", [])
    if ce:
        print(f"\nCAMERA ERRORS / STALLS ({len(ce)})")
        for c in ce[:15]:
            print(f"  {int(c['rel']//60)}:{int(c['rel']%60):02d}  {c.get('message')}")

    # ---- decodes
    res = by.get("result", [])
    if res:
        tot = [r["latency"]["total"] for r in res if r.get("latency")]
        enc = [r["latency"].get("encode") for r in res if r.get("latency") and r["latency"].get("encode") is not None]
        ph = [r["latency"].get("phrase") for r in res if r.get("latency") and r["latency"].get("phrase") is not None]
        early = [r for r in res if r.get("early")]
        print(f"\nDECODES ({len(res)}; {len(early)} committed early while mouthing)")
        print(f"  total      p50 {fmt_ms(pct(tot, .5))}   p95 {fmt_ms(pct(tot, .95))}   max {fmt_ms(max(tot) if tot else None)}")
        if enc: print(f"  encode     p50 {fmt_ms(pct(enc, .5))}   max {fmt_ms(max(enc))}")
        if ph:  print(f"  phrase     p50 {fmt_ms(pct(ph, .5))}   max {fmt_ms(max(ph))}")
        conf = [r.get("confidence") for r in res if r.get("confidence") is not None]
        if conf: print(f"  confidence median {st.median(conf)*100:.0f}%   under 50%: {sum(c < .5 for c in conf)}")
        slow = [r for r in res if r.get("latency", {}).get("total", 0) > 1.0]
        if slow:
            print("  slow (>1 s):", ", ".join(f"{int(r['rel']//60)}:{int(r['rel']%60):02d} {r['latency']['total']:.1f}s {r.get('selected')!r}" for r in slow[:8]))
        print("  last 8:", " | ".join(f"{r.get('selected')} {round((r.get('confidence') or 0)*100)}% {round(r['latency']['total']*1000)}ms" for r in res[-8:]))
    pa = by.get("partial", [])
    if pa:
        sm = [p["score_ms"] for p in pa if p.get("score_ms") is not None]
        print(f"\nSTREAMING PARTIALS ({len(pa)})  score p50 {pct(sm, .5)} ms  p95 {pct(sm, .95)} ms  max {max(sm)} ms")

    # ---- voice
    dv = by.get("delivery", [])
    if dv:
        cached = [d for d in dv if d.get("cached")]
        tt = [d.get("total") for d in dv if d.get("total") is not None]
        print(f"\nVOICE ({len(dv)} deliveries, {len(cached)} cached)  total p50 {pct(tt, .5)} s  max {max(tt) if tt else None} s")
    al = by.get("alert", [])
    if al:
        print(f"ALERTS ({len(al)}): " + ", ".join(a.get("text", "") for a in al[:6]))

    # ---- errors
    er = by.get("error", [])
    if er:
        print(f"\nERRORS ({len(er)})")
        seen = {}
        for e in er:
            seen[e.get("message")] = seen.get(e.get("message"), 0) + 1
        for m, n in sorted(seen.items(), key=lambda x: -x[1])[:10]:
            print(f"  x{n}  {m}")
    me = by.get("monitor_error", [])
    if me:
        print(f"monitor errors: {len(me)} e.g. {me[0].get('error')}")

    if "--events" in sys.argv:
        print("\nTIMELINE")
        for r in rows:
            if r["ev"] in ("camera_error", "error", "result", "alert", "refined", "camera_opened"):
                d = {k: v for k, v in r.items() if k not in ("t", "rel", "ev")}
                print(f"  {int(r['rel']//60)}:{int(r['rel']%60):02d}  {r['ev']:13s} {json.dumps(d)[:160]}")


if __name__ == "__main__":
    main()
