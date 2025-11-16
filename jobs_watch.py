# Adzuna Jobs Digest (multi-profile)
# - per-profile "new" vs preview (jak w watcherach)
# - global de-dup
# - per-profile last_seen & seen_ids w data/jobs_state.json
# - OR po słowach kluczowych via what_or=python,sql,l2
# - wykrywanie EN via lingua-language-detector
# - SUBJECT COUNTERS + CSV ATTACHMENT
# - PUBLISHER: wyprowadzany z przekierowania Adzuny (opcjonalnie)
#
# ENV wymagane:
#   ADZUNA_APP_ID, ADZUNA_APP_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO
#
# ENV opcjonalne:
#   ALWAYS_EMAIL=1|0
#   PREVIEW_LAST_N=10
#   DRY_RUN=0|1
#   RESOLVE_PUBLISHER_REDIRECTS=1|0           (domyślnie 0 = nie śledzimy)
#   RESOLVE_PUBLISHER_MAX=<int>|""            ("" lub brak = bez limitu)
#   RESOLVE_WORKERS=8                         (liczba wątków dla resolvera)
#   RESOLVE_TIMEOUT=6                         (sekundy dla HEAD/GET)

import os, json, smtplib, ssl, math, csv, io, re
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
import requests
from urllib.parse import urlparse
from functools import lru_cache
import concurrent.futures

# ----------- ENV (secrets) -----------
ADZUNA_APP_ID  = os.environ["ADZUNA_APP_ID"]
ADZUNA_APP_KEY = os.environ["ADZUNA_APP_KEY"]

GMAIL_USER         = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
EMAIL_TO           = os.environ["EMAIL_TO"]

# ----------- ENV helpers -----------
def _env_bool(name, default):
    v = os.environ.get(name, None)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes")

def _env_int(name, default):
    v = os.environ.get(name, None)
    if v is None or str(v).strip()=="":
        return default
    try: return int(v)
    except: return default

def _env_int_or_none(name):
    v = os.environ.get(name, None)
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    try: return int(s)
    except: return None

STATE_PATH = Path("data/jobs_state.json")
SETTINGS_PATH = Path("settings.json")
ADZUNA_BASE = "https://api.adzuna.com/v1/api"

EN_NATIVE = {"gb","us","ca","au","nz","ie","sg"}  # rynki natywnie EN

# ----------- IO helpers -----------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_settings():
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        s = json.load(f)
    s["always_email"]   = _env_bool("ALWAYS_EMAIL", s.get("always_email", True))
    s["preview_last_n"] = _env_int("PREVIEW_LAST_N", int(s.get("preview_last_n", 10)))
    s["dry_run"]        = _env_bool("DRY_RUN", s.get("dry_run", False))

    ld = s.get("language_detection") or {}
    s["language_detection"] = {"min_prob": float(ld.get("min_prob", 0.60))}
    return s

def load_state():
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    return {"profiles": {}, "publishers": {}}

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
    R = 6371.0088
    import math as _m
    phi1 = _m.radians(lat1); phi2 = _m.radians(lat2)
    dphi = phi2 - phi1
    dl   = _m.radians(lon2 - lon1)
    a = _m.sin(dphi/2)**2 + _m.cos(phi1)*_m.cos(phi2)*_m.sin(dl/2)**2
    c = 2 * _m.atan2(_m.sqrt(a), _m.sqrt(1-a))
    return R * c

def canon_url(u):
    return (u or "").strip()

def snippet(txt, limit=200):
    if not txt: return ""
    t = re.sub(r"\s+", " ", txt).strip()
    if len(t) <= limit: return t
    return t[:limit-1] + "…"

def _domain(u: str) -> str:
    try:
        h = urlparse(u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""

_ADZ_ID_RX = re.compile(r"/ad/(\d+)")

def adzuna_id_from_url(u: str) -> str | None:
    m = _ADZ_ID_RX.search(u or "")
    return m.group(1) if m else None

# ----------- language detection (lingua) -----------
from lingua import Language, LanguageDetectorBuilder
_LANGS = [
    Language.ENGLISH, Language.POLISH, Language.GERMAN, Language.FRENCH, Language.DUTCH,
    Language.ITALIAN, Language.SPANISH, Language.PORTUGUESE,
    Language.SWEDISH, Language.DANISH, Language.FINNISH,
    Language.CZECH, Language.SLOVAK, Language.ROMANIAN, Language.HUNGARIAN
]
_DETECTOR = LanguageDetectorBuilder.from_languages(*_LANGS).build()

def looks_english(text: str, min_prob: float = 0.60) -> bool:
    if not text:
        return False
    t = text[:10000]
    try:
        lang = _DETECTOR.detect_language_of(t)
        if lang == Language.ENGLISH:
            conf = _DETECTOR.compute_language_confidence(t, Language.ENGLISH)
            if conf >= min_prob:
                return True
    except Exception:
        pass
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

# ----------- publisher resolver -----------
RESOLVE_PUBLISHER = _env_bool("RESOLVE_PUBLISHER_REDIRECTS", False)
RESOLVE_MAX = _env_int_or_none("RESOLVE_PUBLISHER_MAX")  # None = bez limitu
RESOLVE_WORKERS = _env_int("RESOLVE_WORKERS", 12)
RESOLVE_TIMEOUT = _env_int("RESOLVE_TIMEOUT", 6)

@lru_cache(maxsize=4096)
def resolve_final_host(u: str) -> str:
    try:
        r = requests.head(u, allow_redirects=True, timeout=RESOLVE_TIMEOUT,
                          headers={"User-Agent":"jobs-watch/1.0"})
        h = _domain(r.url)
        if h: return h
    except Exception:
        pass
    try:
        r = requests.get(u, allow_redirects=True, timeout=RESOLVE_TIMEOUT, stream=True,
                         headers={"User-Agent":"jobs-watch/1.0"})
        h = _domain(r.url)
        if h: return h
    except Exception:
        pass
    return _domain(u)

# ----------- normalize & filters -----------
def normalize_job_adzuna(j, country_code):
    created = iso_to_dt(j.get("created"))
    comp = (j.get("company") or {}).get("display_name") or ""
    loc  = (j.get("location") or {}).get("display_name") or ""
    url  = canon_url(j.get("redirect_url") or "")
    lat  = j.get("latitude")
    lon  = j.get("longitude")
    publisher = _domain(url)  # baseline (może zostać nadpisany po resolverze)

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
        "publisher": publisher,
        "desc": j.get("description") or "",
        "lat": lat,
        "lon": lon,
        "country": country_code
    }

def passes_filters(job, prof, lang_min_prob: float):
    title = (job["title"] or "").lower()
    comp  = (job["company"] or "").lower()
    desc  = (job["desc"] or "").lower()
    text  = " ".join([title, comp, desc])

    inc = prof.get("keywords_any") or []
    if inc and not any(k.lower() in text for k in inc):
        return False

    exc = [x.lower() for x in (prof.get("keywords_exclude") or [])]
    if exc and any(x in text for x in exc):
        return False

    if prof.get("remote_only") and not is_remote_like(text):
        return False

    if prof.get("english_only") and job["country"] not in EN_NATIVE:
        if not looks_english((job["title"] or "") + " " + (job["desc"] or ""), min_prob=lang_min_prob):
            return False

    ms = prof.get("min_salary")
    if ms is not None:
        try:
            ms = float(ms)
            if (job["salary_min"] or 0) < ms and (job["salary_max"] or 0) < ms:
                return False
        except Exception:
            pass

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

    where = (prof.get("location") or "").strip()
    if where and where.lower() not in job["location"].lower():
        if not (prof.get("remote_only") and is_remote_like(text)):
            return False

    return True

# ----------- fetch -----------
def fetch_adzuna_page(country, page, prof):
    keywords_any = prof.get("keywords_any") or []
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": int(prof.get("results_per_page") or 50),
        "sort_by": "date",
        "content-type": "application/json",
    }
    if keywords_any:
        params["what_or"] = ",".join(keywords_any)
    where = (prof.get("location") or "").strip()
    if where:
        params["where"] = where

    url = f"{ADZUNA_BASE}/jobs/{country}/search/{page}"
    r = requests.get(url, params=params, timeout=25)
    if r.status_code == 404:
        print(f"[jobs] {prof['name']} country={country} page={page} SKIP: 404")
        return []
    r.raise_for_status()
    return r.json().get("results", []) or []

def fetch_profile_pool(prof, lang_min_prob: float):
    if prof.get("country"):
        countries = [prof["country"]]
    else:
        countries = prof.get("countries") or ["pl"]

    merged, seen = [], set()
    for c in countries:
        max_pages = int(prof.get("max_pages") or 2)
        for p in range(1, max_pages + 1):
            try:
                rows = fetch_adzuna_page(c, p, prof)
            except requests.HTTPError as e:
                print(f"[jobs] {prof['name']} country={c} page={p} ERROR: {repr(e)}")
                break
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
        pub = it.get("publisher") or ""
        pub_html = f" · <span style='color:#777'>{pub}</span>" if pub else ""
        parts.append(
            "<li style='margin:0 0 14px'>"
            f"<div style='font-weight:600'>{link}</div>"
            f"<div style='color:#555;font-size:13px'>{it['company']} · {it['location']} · {when}{salary_html} · "
            f"<span style='color:#777'>{it['source']}</span>{pub_html}</div>"
            "</li>"
        )
    parts.append("</ul>")
    return "".join(parts)

def build_subject(sections_info, always_email):
    total_new = sum(s["new_count"] for s in sections_info)
    if total_new == 0 and always_email:
        return "[Jobs] No new — preview only"
    if total_new == 0:
        return "[Jobs] No new today"
    entries = []
    for s in sections_info:
        if s["new_count"] > 0:
            entries.append(f"{s['name']}: {s['new_count']} new")
        else:
            entries.append(f"{s['name']}: preview")
    subj_core = " • ".join(entries)
    if len(subj_core) > 180:
        subj_core = subj_core[:177] + "…"
    return f"[Jobs] {subj_core}"

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

def build_csv(all_sections_with_flags):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "section","is_new","created_at","title","company","location","country",
        "salary_min","salary_max","source","publisher","url","remote_detected","english_detected",
        "uid","desc_snippet"
    ])
    for name, items, is_preview, new_uids in all_sections_with_flags:
        for it in items:
            is_new = (it["uid"] in new_uids)
            created = it["created_at"].isoformat() if it["created_at"] else ""
            txt = (it.get("title","") + " " + it.get("desc","")).strip()
            eng = looks_english(txt, 0.60)
            rem = is_remote_like(txt)
            w.writerow([
                name,
                "true" if is_new else "false",
                created,
                it["title"],
                it["company"],
                it["location"],
                it["country"],
                it["salary_min"] or "",
                it["salary_max"] or "",
                it["source"],
                it.get("publisher",""),
                it["url"],
                "true" if rem else "false",
                "true" if eng else "false",
                it["uid"],
                snippet(it.get("desc",""))
            ])
    return buf.getvalue()

def send_email(subject_text, html_body, csv_text, csv_filename):
    msg = EmailMessage()
    msg["Subject"] = subject_text
    msg["From"] = GMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content("Daily jobs digest attached as CSV.\n(Open the HTML part for a nicer view.)")
    msg.add_alternative(html_body, subtype="html")
    if csv_text:
        # ważne: bez 'maintype=' dla string payload
        msg.add_attachment(csv_text, subtype="csv", filename=csv_filename)
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
    pub_cache = state.setdefault("publishers", {})

    searches = settings.get("searches") or []
    always_email = settings.get("always_email", True)
    preview_n = int(settings.get("preview_last_n", 10))
    dry_run = bool(settings.get("dry_run", False))
    lang_min_prob = float((settings.get("language_detection") or {}).get("min_prob", 0.60))

    global_seen_uids = set()
    sections_basic = []             # (name, items, is_preview)
    sections_with_flags = []        # (name, items, is_preview, new_uid_set)
    sections_info = []              # {name, new_count, is_preview}

    now_utc = datetime.now(timezone.utc)

    # --- fetch & decide ---
    for prof in searches:
        name = prof["name"]
        prof_state = profiles_state.setdefault(name, {"last_seen": None, "seen_ids": []})
        last_seen = datetime.fromisoformat(prof_state["last_seen"]) if prof_state.get("last_seen") else None
        seen_ids = set(prof_state.get("seen_ids", []))

        items = fetch_profile_pool(prof, lang_min_prob)

        # global de-dup
        deduped = []
        for it in items:
            if it["uid"] in global_seen_uids:
                continue
            global_seen_uids.add(it["uid"])
            deduped.append(it)
        items = deduped

        # new vs preview
        if last_seen:
            candidates = [
                it for it in items
                if (it["created_at"] and it["created_at"] > last_seen) or (not it["created_at"] and it["uid"] not in seen_ids)
            ]
        else:
            candidates = []

        run_seen, new_items = set(), []
        for it in candidates:
            uid = it["uid"]
            if not uid or uid in seen_ids or uid in run_seen:
                continue
            run_seen.add(uid)
            new_items.append(it)

        if new_items:
            sections_basic.append((name, new_items, False))
            new_uid_set = {it["uid"] for it in new_items}
            sections_with_flags.append((name, new_items, False, new_uid_set))
            sections_info.append({"name": name, "new_count": len(new_items), "is_preview": False})
        else:
            preview = items[:max(preview_n, 0)]
            sections_basic.append((name, preview, True))
            sections_with_flags.append((name, preview, True, set()))
            sections_info.append({"name": name, "new_count": 0, "is_preview": True})

        # watermark → newest fetched
        newest_ts = max((it["created_at"] for it in items if it["created_at"]), default=last_seen)
        if newest_ts and newest_ts > now_utc:
            newest_ts = now_utc
        if newest_ts:
            prof_state["last_seen"] = newest_ts.isoformat()

        # persist seen_ids dla "new"
        for it in new_items:
            if it["uid"]:
                seen_ids.add(it["uid"])
        prof_state["seen_ids"] = list(seen_ids)[-50000:]

    # --- resolve publishers for items we're emailing (new + preview) ---
    if RESOLVE_PUBLISHER:
        to_resolve = []
        # zbierz wszystkie itemy do wysłania
        emailing_items = []
        for _, items, _ in sections_basic:
            emailing_items.extend(items)

        for it in emailing_items:
            url = it.get("url") or ""
            if not url:
                continue
            adz_id = adzuna_id_from_url(url)
            # jeśli już mamy finalną domenę zapisaną wcześniej
            if adz_id and adz_id in pub_cache:
                it["publisher"] = pub_cache[adz_id]
                continue
            # jeśli baseline jest sensowny (nie adzuna), nie musimy rozwiązywać
            if it.get("publisher") and "adzuna" not in it["publisher"]:
                continue
            if adz_id:
                to_resolve.append((adz_id, url, it))

        # limit (None => bez limitu)
        if RESOLVE_MAX is not None:
            to_resolve = to_resolve[:RESOLVE_MAX]

        if to_resolve:
            with concurrent.futures.ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as ex:
                futs = {ex.submit(resolve_final_host, url): (adz_id, it) for adz_id, url, it in to_resolve}
                for fut, (adz_id, it) in futs.items():
                    try:
                        host = fut.result()
                        if host:
                            it["publisher"] = host
                            if adz_id:
                                pub_cache[adz_id] = host
                    except Exception:
                        pass

    # --- subject, body, csv, send ---
    subject_counters = build_subject(sections_info, always_email)
    tagline = ", ".join(p["name"] for p in searches)
    _, _, html_body = build_email(sections_basic, tagline, always_email)
    csv_text = build_csv(sections_with_flags)
    csv_filename = f"jobs_{now_utc.date().isoformat()}.csv"

    if not dry_run:
        send_email(subject_counters, html_body, csv_text, csv_filename)

    # zapisz updated state (w tym publishers cache)
    state["publishers"] = pub_cache
    save_state(state)

    counts = ", ".join([f"{s['name']}: {s['new_count']}{'' if not s['is_preview'] else 'P'}" for s in sections_info])
    print(f"[jobs] done: sections=({counts}), dry_run={dry_run}")

if __name__ == "__main__":
    main()
