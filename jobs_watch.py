# Adzuna Jobs Digest (multi-profile) — mirrors your watcher pattern
# - per-profile "new" vs preview
# - global de-dup across profiles
# - per-profile last_seen & seen_ids in data/jobs_state.json
# - optional env overrides for ALWAYS_EMAIL / PREVIEW_LAST_N / DRY_RUN
# - english detection via lingua-language-detector (pure Python)

import os, json, smtplib, ssl, math
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
import requests

# ----------- ENV (secrets) -----------
ADZUNA_APP_ID  = os.environ["ADZUNA_APP_ID"]
ADZUNA_APP_KEY = os.environ["ADZUNA_APP_KEY"]

GMAIL_USER         = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
EMAIL_TO           = os.environ["EMAIL_TO"]

# optional env overrides
def _env_bool(name, default):
    v = os.environ.get(name, None)
    if v is None: return default
    return str(v).strip() in ("1","true","True","yes","YES")

def _env_int(name, default):
    v = os.environ.get(name, None)
    if v is None or str(v).strip()=="":
        return default
    try: return int(v)
    except: return default

STATE_PATH = Path("data/jobs_state.json")
SETTINGS_PATH = Path("settings.json")
ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"

EN_NATIVE = {"gb","us","ca","au","nz","ie","sg"}  # anglojęzyczne

# ----------- IO helpers -----------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_settings():
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        s = json.load(f)

    # env overrides (optional knobs)
    s["always_email"]   = _env_bool("ALWAYS_EMAIL", s.get("always_email", True))
    s["preview_last_n"] = _env_int("PREVIEW_LAST_N", int(s.get("preview_last_n", 10)))
    s["dry_run"]        = _env_bool("DRY_RUN", s.get("dry_run", False))

    # language detection config (optional)
    ld = s.get("language_detection") or {}
    s["language_detection"] = {
        "min_prob": float(ld.get("min_prob", 0.60))
    }
    return s

def load_state():
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    return {"profiles": {}}

def save_state(st):
    ensure_dir(STATE_PATH)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

# ----------- utils -----------
def iso_to_dt(s):
    if not s: return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)
        except Exception:
            return None

def km_distance(lat1, lon1, lat2, lon2):
    # Haversine
    R = 6371.0088
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def canon_url(u):
    return (u or "").strip().lower()

# ----------- language detection (lingua-language-detector) -----------
from lingua import Language, LanguageDetectorBuilder

# Keep a focused set (removed Norwegian to avoid enum mismatch)
_LANGS = [
    Language.ENGLISH, Language.POLISH, Language.GERMAN, Language.FRENCH, Language.DUTCH,
    Language.ITALIAN, Language.SPANISH, Language.PORTUGUESE,
    Language.SWEDISH, Language.DANISH, Language.FINNISH,
    Language.CZECH, Language.SLOVAK, Language.ROMANIAN, Language.HUNGARIAN
]
_DETECTOR = LanguageDetectorBuilder.from_languages(*_LANGS).build()

def looks_english(text: str, min_prob: float = 0.60) -> bool:
    """Detect EN with Lingua; fallback to tiny heuristic on errors."""
    if not text:
        return False
    t = text[:10000]
    try:
        lang = _DETECTOR.detect_language_of(t)
        if lang == Language.ENGLISH:
            conf = _DETECTOR.compute_language_confidence(t, Language.ENGLISH)  # 0..1
            if conf >= min_prob:
                return True
    except Exception:
        pass

    # minimal fallback if detector errors (rare)
    tl = t.lower()
    kw = ["remote","engineer","developer","requirements","responsibilities","benefits",
          "apply","experience","position","stack","team","salary","usd","eur",
          "hybrid","onsite","python","sql","support","incident","cloud","devops"]
    hits = sum(1 for k in kw if k in tl)
    ascii_ratio = sum(ch.isascii() for ch in tl) / max(1, len(tl))
    return ascii_ratio > 0.85 or hits >= 2

def is_remote_like(text: str) -> bool:
    if not text: return False
    t = text.lower()
    keys = ["remote", "fully remote", "work from home", "wfh", "home office",
            "praca zdalna", "zdalnie", "100% zdalnie", "zdalna"]
    return any(k in t for k in keys)

def normalize_job_adzuna(j, country_code):
    created = iso_to_dt(j.get("created"))
    comp = (j.get("company") or {}).get("display_name") or ""
    loc  = (j.get("location") or {}).get("display_name") or ""
    url  = canon_url(j.get("redirect_url") or "")
    lat  = j.get("latitude")
    lon  = j.get("longitude")
    return {
        "uid": str(j.get("id") or url or f"{j.get('title','')}|{comp}|{loc}"),
        "title": j.get("title") or "",
        "company": comp,
        "location": loc,
        "created_at": created,
        "salary_min": j.get("salary_min"),
        "salary_max": j.get("salary_max"),
        "url": url,
        "source": f"Adzuna/{country_code}",
        "desc": j.get("description") or "",
        "lat": lat,
        "lon": lon,
        "country": country_code
    }

def passes_filters(job, prof, lang_min_prob: float):
    # keywords include/exclude
    title = job["title"].lower()
    comp  = job["company"].lower()
    desc  = (job["desc"] or "").lower()
    text  = " ".join([title, comp, desc])

    inc = prof.get("keywords_any") or []
    if inc:
        if not any(k.lower() in text for k in inc):
            return False

    exc = [x.lower() for x in (prof.get("keywords_exclude") or [])]
    if exc and any(x in text for x in exc):
        return False

    # remote_only
    if prof.get("remote_only"):
        if not is_remote_like(text):
            return False

    # english_only (strict for non-native English markets)
    if prof.get("english_only"):
        if job["country"] in EN_NATIVE:
            pass
        else:
            if not looks_english((job["title"] or "") + " " + (job["desc"] or ""), min_prob=lang_min_prob):
                return False

    # min_salary
    ms = prof.get("min_salary")
    if ms is not None:
        try:
            ms = float(ms)
            if (job["salary_min"] or 0) < ms and (job["salary_max"] or 0) < ms:
                return False
        except Exception:
            pass

    # radius
    center = prof.get("center")
    rk = prof.get("radius_km")
    if center and rk:
        try:
            rk = float(rk)
            if job["lat"] is None or job["lon"] is None:
                return False
            d = km_distance(center["lat"], center["lng"], float(job["lat"]), float(job["lon"]))
            if d > rk:
                return False
        except Exception:
            pass

    # location (lekko – Adzuna i tak filtruje po 'where')
    where = (prof.get("location") or "").strip()
    if where:
        if where.lower() not in job["location"].lower():
            if not (prof.get("remote_only") and is_remote_like(text)):
                return False

    return True

# ----------- fetch -----------
def fetch_adzuna_page(country, page, prof):
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": int(prof.get("results_per_page") or 50),
        "what": ",".join(prof.get("keywords_any") or []),
        "where": prof.get("location") or "",
        "sort_by": "date",
        "content-type": "application/json",
    }
    url = f"{ADZUNA_BASE}/{country}/search/{page}"
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json().get("results", []) or []

def fetch_profile_pool(prof, lang_min_prob: float):
    if prof.get("country"):
        countries = [prof["country"]]
    else:
        countries = prof.get("countries") or ["pl"]

    merged = []
    seen = set()
    for c in countries:
        max_pages = int(prof.get("max_pages") or 2)
        for p in range(1, max_pages + 1):
            try:
                rows = fetch_adzuna_page(c, p, prof)
            except Exception as e:
                print(f"[jobs] {prof['name']} country={c} page={p} ERROR: {repr(e)}")
                break
            print(f"[jobs] {prof['name']} country={c} page={p} results={len(rows)}")
            if not rows:
                break
            for j in rows:
                nj = normalize_job_adzuna(j, c)
                if passes_filters(nj, prof, lang_min_prob):
                    uid = nj["uid"]
                    if uid not in seen:
                        seen.add(uid)
                        merged.append(nj)

    merged.sort(key=lambda it: it["created_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return merged

# ----------- email -----------
def build_section_html(title, items, is_preview):
    parts = []
    parts.append(f"<h3 style='margin:18px 0 8px'>{title}{' (preview)' if is_preview else ''}</h3>")
    if not items:
        parts.append("<p style='color:#666;margin:0 0 8px'>No items.</p>")
        return "".join(parts)

    parts.append("<ul style='list-style:none;margin:0;padding:0'>")
    for it in items:
        when = it["created_at"].strftime("%Y-%m-%d %H:%M UTC") if it["created_at"] else ""
        salary = ""
        if it["salary_min"] or it["salary_max"]:
            a = int(it["salary_min"] or 0); b = int(it["salary_max"] or 0)
            if a and b: salary = f"{a:,}–{b:,}".replace(",", " ")
            elif a:     salary = f"{a:,}".replace(",", " ")
            elif b:     salary = f"{b:,}".replace(",", " ")
        salary_html = f" · <span style='color:#155'>{salary}</span>" if salary else ""
        link = f"<a href='{it['url']}' style='color:#0b57d0;text-decoration:none'>{it['title']}</a>" if it["url"] else it["title"]
        parts.append(
            "<li style='margin:0 0 14px'>"
            f"<div style='font-weight:600'>{link}</div>"
            f"<div style='color:#555;font-size:13px'>{it['company']} · {it['location']} · {when}{salary_html} · <span style='color:#777'>{it['source']}</span></div>"
            "</li>"
        )
    parts.append("</ul>")
    return "".join(parts)

def build_email(all_sections, subject_suffix, always_email):
    any_items = any(len(items)>0 for _, items, _ in all_sections)
    subject = f"[Jobs] Daily digest — {subject_suffix}"
    if not any_items and not always_email:
        subject = "[Jobs] No new today"

    parts = []
    parts.append("<!doctype html><html><body style='font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;font-size:14px;color:#111;margin:0;padding:16px'>")
    parts.append(f"<h2 style='margin:0 0 10px'>Daily Jobs Digest</h2>")
    for name, items, is_preview in all_sections:
        parts.append(build_section_html(name, items, is_preview))
    if not any_items and always_email:
        parts.append("<p style='color:#666;margin-top:10px'>No new jobs today — showing empty sections.</p>")
    parts.append("</body></html>")
    html_body = "".join(parts)

    plain = []
    if any_items:
        for name, items, is_preview in all_sections:
            plain.append(f"{name}{' (preview)' if is_preview else ''}: {len(items)}")
            for it in items:
                when = it["created_at"].strftime("%Y-%m-%d %H:%M UTC") if it["created_at"] else ""
                line = f"- {it['title']} — {it['company']} · {it['location']} · {when}"
                if it["url"]: line += f" · {it['url']}"
                plain.append(line)
            plain.append("")
    else:
        plain.append("No new jobs today.")
    return subject, "\n".join(plain), html_body

def send_email(items_sections, subject_suffix, always_email):
    subject, plain_text, html_body = build_email(items_sections, subject_suffix, always_email)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content(plain_text)
    msg.add_alternative(html_body, subtype="html")
    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls(context=ctx)
        smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

# ----------- main -----------
def main():
    settings = load_settings()
    state = load_state()
    profiles_state = state.setdefault("profiles", {})

    searches = settings.get("searches") or []
    always_email = settings.get("always_email", True)
    preview_n = int(settings.get("preview_last_n", 10))
    dry_run = bool(settings.get("dry_run", False))
    lang_min_prob = float((settings.get("language_detection") or {}).get("min_prob", 0.60))

    global_seen_uids = set()
    sections = []
    now_utc = datetime.now(timezone.utc)

    for prof in searches:
        name = prof["name"]
        prof_state = profiles_state.setdefault(name, {"last_seen": None, "seen_ids": []})
        last_seen = datetime.fromisoformat(prof_state["last_seen"]) if prof_state.get("last_seen") else None
        seen_ids = set(prof_state.get("seen_ids", []))

        items = fetch_profile_pool(prof, lang_min_prob)

        deduped = []
        for it in items:
            if it["uid"] in global_seen_uids:
                continue
            global_seen_uids.add(it["uid"])
            deduped.append(it)
        items = deduped

        if last_seen:
            candidates = [
                it for it in items
                if (it["created_at"] and it["created_at"] > last_seen) or (not it["created_at"] and it["uid"] not in seen_ids)
            ]
        else:
            candidates = []  # seed → preview only

        run_seen, new_items = set(), []
        for it in candidates:
            uid = it["uid"]
            if not uid or uid in seen_ids or uid in run_seen:
                continue
            run_seen.add(uid)
            new_items.append(it)

        if new_items:
            sections.append((name, new_items, False))
        else:
            preview = items[:max(preview_n, 0)]
            sections.append((name, preview, True))

        newest_ts = max((it["created_at"] for it in items if it["created_at"]), default=last_seen)
        if newest_ts and newest_ts > now_utc:
            newest_ts = now_utc
        if newest_ts:
            prof_state["last_seen"] = newest_ts.isoformat()

        for it in new_items:
            if it["uid"]:
                seen_ids.add(it["uid"])
        prof_state["seen_ids"] = list(seen_ids)[-50000:]

    if not dry_run:
        tagline = ", ".join(p["name"] for p in searches)
        send_email(sections, tagline, always_email)

    save_state(state)

    counts = ", ".join([f"{name}: {len(items)}{'P' if is_prev else 'N'}" for name, items, is_prev in sections])
    print(f"[jobs] done: sections=({counts}), dry_run={dry_run}")

if __name__ == "__main__":
    main()
