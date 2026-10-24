import asyncio, hashlib, json, os, re, sqlite3, threading, time, uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

# ---------- config ----------
# Set BASE_URL in Render to EXACTLY what you submit, e.g. https://ga3-2.onrender.com/a2a/
BASE_URL = os.environ.get("BASE_URL") or (os.environ["RENDER_EXTERNAL_URL"].rstrip("/") + "/a2a/")
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
API_BASE = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openrouter/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4.1")
REASONING_MODEL = bool(re.search(r"(^|/)(gpt-5|o\d)", MODEL))   # these reject temperature / max_tokens
CHUNK = int(os.environ.get("CHUNK", "1"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "25"))
DECIDE_BUDGET = float(os.environ.get("DECIDE_BUDGET", "38"))

BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MT = "application/vnd.ga5.invoice-action-results+json"
REC_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}

# ---------- storage ----------
db = sqlite3.connect(os.environ.get("DB_PATH", "a2a.db"), check_same_thread=False)
db.executescript("""
create table if not exists tasks(id text primary key, principal text, created real, data text);
create table if not exists msgs(principal text, mid text, hash text, task_id text, primary key(principal, mid));
create table if not exists cache(k text primary key, v text);
""")
dblock = threading.Lock()

def q1(sql, args=()):
    with dblock:
        return db.execute(sql, args).fetchone()

def qall(sql, args=()):
    with dblock:
        return db.execute(sql, args).fetchall()

def save_task(t, principal, mrec=None):
    with dblock:
        db.execute("insert into tasks values(?,?,?,?) on conflict(id) do update set data=excluded.data",
                   (t["id"], principal, time.time(), json.dumps(t)))
        if mrec:
            db.execute("insert into msgs values(?,?,?,?)", mrec)
        db.commit()

def load_task(tid, principal):
    r = q1("select data from tasks where id=? and principal=?", (tid, principal))
    return json.loads(r[0]) if r else None

def cache_get(k):
    r = q1("select v from cache where k=?", (k,))
    return json.loads(r[0]) if r else None

def cache_put(k, v):
    with dblock:
        db.execute("insert or replace into cache values(?,?)", (k, json.dumps(v)))
        db.commit()

# ---------- helpers ----------
def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def flat(o):
    if isinstance(o, str): return o
    if isinstance(o, dict): return "\n\n".join(flat(v) for v in o.values())
    if isinstance(o, list): return "\n\n".join(flat(v) for v in o)
    return str(o)

def out(status, obj):
    return Response(json.dumps(obj, separators=(",", ":")).encode(), status_code=status,
                    media_type="application/a2a+json")

def err(status, code, msg):
    return out(status, {"error": {"code": code, "message": msg}})

locks = defaultdict(asyncio.Lock)

def gate(req, post_send=False):
    m = re.fullmatch(r"Bearer (\S+)", req.headers.get("authorization", ""))
    if not m:
        return None, err(401, "UNAUTHENTICATED", "Authentication required")
    if req.headers.get("a2a-version") != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported protocol version")
    if post_send and not req.headers.get("content-type", "").lower().startswith("application/a2a+json"):
        return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json")
    return sha(m.group(1)), None

def uniq(seq):
    seen, o = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); o.append(x)
    return o

# ---------- document analysis (deterministic) ----------
REF = re.compile(r"\[[^\[\]\n]+\]")
CACHE_V = "v7:"   # bump to invalidate old cached decisions

CURS = "INR|USD|EUR|GBP|JPY|KRW|AED|SGD|AUD|CAD|CHF|CNY|HKD|SAR|ZAR|NZD|SEK|NOK|DKK"
AMT = re.compile(rf"\b({CURS})\s?(\d[\d,]*(?:\.\d+)?)")
ZERO_DEC = {"JPY", "KRW"}
COVER = re.compile(rf"Supplier (.+?); invoice (\S+?); stated total ({CURS}) (\d[\d,]*(?:\.\d+)?)")
INVNO = re.compile(r"\bINV-[A-Za-z0-9-]*\d")
VENDOR = re.compile(r"\bSupplier ([A-Z][A-Za-z0-9&'.\- ]{1,60}?)(?=;|,|\.\s| invoice| covers| for | billed| issued|$)")
DECOY = re.compile(r"training|archiv|historical|expired|superseded|separate supplier|different (order|supplier|vendor|invoice)"
                   r"|unrelated|hypothetical|not this|no live|govern no|previous quarter|closed (reconciliation )?incident"
                   r"|similar trading name|example|cover sheet", re.I)
NEG = re.compile(r"\b(no|not|never|without|nor)\b", re.I)
LIMIT_WORDS = re.compile(r"authority|delegat|limit|threshold|ceiling|escalat|approver|approval", re.I)

SIG = {
    "reject_duplicate": re.compile(
        r"already (been )?(paid|discharged|settled)|completed payment|previously (paid|settled)|paid in full"
        r"|requires rejection|same commercial key|duplicate payment", re.I),
    "open_exception": re.compile(
        r"outside (the )?(permitted |agreed |stated )?(reconciliation )?tolerance|contradict|exception (case|process|workflow)"
        r"|conflicting (signed )?records|does not (match|agree|reconcile)|short[- ]?shipped|variance exceeds", re.I),
    "hold_invoice": re.compile(
        r"awaiting (independent )?(verification|callback|confirmation)|callback (is |remains )?(pending|outstanding)"
        r"|pending (independent )?verification|pauses? payment|payment (is |remains )?paused|unverified bank"
        r"|until (the )?\w+ (verification|check|callback) (completes|is complete)", re.I),
    "request_approval": re.compile(
        r"outside (the )?(operator|queue|delegat|autonomous)|exceeds? (the )?(\w+ ){0,3}(authority|limit|ceiling|threshold)"
        r"|above (the )?(\w+ ){0,3}(authority|limit|ceiling|threshold)|named approver|financial-approval"
        r"|approval workflow|beyond (\w+ ){0,3}(authority|limit)", re.I),
}
PRIORITY = ["reject_duplicate", "open_exception", "hold_invoice", "request_approval"]
CLEAN = re.compile(
    r"agree within|within (the )?(stated |permitted )?(rounding )?tolerance|three-way match|reconcil(es|ed)\b"
    r"|same payable total|full quantity|covers every billed unit|accepted-goods|no earlier posting|no paid item"
    r"|no (prior|previous) payment|within (the )?(\w+ ){0,3}(authority|limit)|bank details? (match|unchanged|verified)"
    r"|callback (was |has been )?(completed|verified)", re.I)

def clean_count(text):
    return len(CLEAN.findall(text))

def minor(cur, s):
    try:
        return int(Decimal(s.replace(",", "")) * (1 if cur in ZERO_DEC else 100))
    except (InvalidOperation, ValueError):
        return 0

def signals(text, total=0):
    hits = set()
    for act, rx in SIG.items():
        for m in rx.finditer(text):
            st = max(text.rfind(".", 0, m.start()), text.rfind(";", 0, m.start())) + 1
            window = text[max(st, m.start() - 50):m.start()]
            if not NEG.search(window):
                hits.add(act); break
    if total and LIMIT_WORDS.search(text):
        for m in AMT.finditer(text):
            v = minor(m.group(1), m.group(2))
            if 0 < v < total:
                hits.add("request_approval"); break
    return hits

def top(hits):
    for a in PRIORITY:
        if a in hits: return a
    return None

def paras(pk):
    # split on blank lines; if a block holds more than 3 refs, split further by line, then by sentence
    blocks = [b.strip() for b in re.split(r"\n\s*\n", flat(pk)) if b.strip()]
    lines = []
    for b in blocks:
        if len(uniq(REF.findall(b))) > 3:
            lines += [l.strip() for l in b.split("\n") if l.strip()]
        else:
            lines.append(b)
    final = []
    for x in lines:
        if len(uniq(REF.findall(x))) > 3:
            final += [s.strip() for s in re.split(r"(?<=[.;])\s+", x) if s.strip()]
        else:
            final.append(x)
    return final

NOISE = re.compile(r"(The referenced office was [^.]*? and the audit export sequence was \d+|Review location [^;]*; docket \d+)\.?\s*$")

def keep(t):  # drop pure boilerplate lines that carry no reference
    return bool(REF.search(t)) or not NOISE.search(t)

def extract_facts(ps):
    text = "\n\n".join(ps)
    m = COVER.search(text)
    if m:
        cur = m.group(3)
        return {"vendorName": m.group(1).strip(), "invoiceNumber": m.group(2).rstrip(".,;"),
                "amountMinor": minor(cur, m.group(4)), "currency": cur}
    f = {}
    for p in ps:                       # live paragraph naming the invoice and its amount
        if DECOY.search(p): continue
        inv, amt = INVNO.search(p), AMT.search(p)
        if inv and amt:
            f = {"invoiceNumber": inv.group(0), "amountMinor": minor(amt.group(1), amt.group(2)), "currency": amt.group(1)}
            break
    for p in ps:
        if DECOY.search(p) and "cover sheet" not in p: continue
        v = VENDOR.search(p)
        if v:
            f["vendorName"] = v.group(1).strip(); break
    return f

def analyze(pk):
    ps = paras(pk)
    text = "\n\n".join(ps)
    refs_of = [uniq(REF.findall(p)) for p in ps]
    tri = [i for i, r in enumerate(refs_of) if len(r) == 3]
    tri_live = [i for i in tri if not DECOY.search(ps[i])] or tri
    facts = extract_facts(ps)
    return {"ps": ps, "text": text, "refs_of": refs_of, "tri": tri_live, "facts": facts}


# ---------- structured solver for multi-paragraph (layout b) packages ----------
PO = r"(PO-[A-Za-z0-9-]*\d)"
AMTG = rf"({CURS})\s?(\d[\d,]*(?:\.\d+)?)"

def _amt(m, i=1):
    return (m.group(i), minor(m.group(i), m.group(i + 1))) if m else (None, 0)

def solve_b(live_text, live_idx, ps):
    T = live_text
    info, used = {}, set()
    def mark(rx):
        for i in live_idx:
            if re.search(rx, ps[i], re.I): used.add(i)
    # current order / invoice amounts
    cur_po = None
    for rx in [rf"currently submitted order is {PO}", rf"Order {PO} is signed", rf"covers order {PO}",
               rf"submitted (?:order|invoice) (?:is |for )?{PO}", rf"current(?:ly)? order {PO}"]:
        m = re.search(rx, T, re.I)
        if m: cur_po = m.group(1); break
    signed_cur, signed = _amt(re.search(rf"signed (?:for|at) {AMTG}", T, re.I))
    inv_cur, inv_total = _amt(re.search(rf"(?:submitted )?invoice total (?:is|of) {AMTG}", T, re.I))
    corr = re.search(rf"(?<!No )corrected invoice[^.]*?{AMTG}", T, re.I)
    if corr: inv_cur, inv_total = _amt(corr)
    inst_cur, inst_total = _amt(re.search(rf"(?:instrument|invoice) INV-[\w-]*[^.]*?{AMTG}", T, re.I))
    total = inv_total or inst_total or signed
    cur = inv_cur or inst_cur or signed_cur
    # prior payment
    prior = re.search(rf"(?:previously|already) (?:settled|paid|discharged) order {PO}", T, re.I) \
        or re.search(rf"settled order {PO}", T, re.I)
    dup = bool(prior and cur_po and prior.group(1).upper() == cur_po.upper())
    if prior: mark(r"(previously|already) (settled|paid|discharged)|settled order")
    # bank change
    bank = re.search(r"Vendor master lists account ([\w-]+)\.? Remittance instructions now specify ([\w-]+)", T, re.I)
    hold = False
    if bank:
        mark(r"Vendor master lists account")
        changed = bank.group(1).upper() != bank.group(2).upper()
        cb = re.search(r"callback[^.]*\.", T[bank.end():], re.I)
        cbt = cb.group(0) if cb else ""
        verified = bool(re.search(r"confirmed|verified|completed", cbt, re.I)) and not re.search(r"awaiting|pending|not yet|unverified|outstanding", cbt, re.I)
        hold = changed and not verified
    # order/invoice difference
    exc = False
    if signed and (inv_total or inst_total):
        other = inv_total or inst_total
        exc = abs(signed - other) > max(100, 0.005 * signed)
        mark(r"signed (for|at)|invoice total")
    if cur_po: mark(re.escape(cur_po))
    # delegation limit
    limit = 0
    for i in live_idx:
        p = ps[i]
        if LIMIT_WORDS.search(p) and not re.search(r"Payment controls", p, re.I):
            for m in AMT.finditer(p):
                v = minor(m.group(1), m.group(2))
                if v and v != total and v != signed:
                    limit = v; used.add(i); break
        if limit: break
    policy_i = [i for i in live_idx if re.search(r"Payment controls|controls?:", ps[i], re.I)]
    used.update(policy_i)
    structured = bool(cur_po or bank or signed or prior)
    if dup: action = "reject_duplicate"
    elif exc: action = "open_exception"
    elif hold: action = "hold_invoice"
    elif limit and total: action = "request_approval" if total > limit else "settle_invoice"
    else: action = None
    info.update(total=total, cur=cur, cur_po=cur_po, limit=limit)
    return action if structured else None, info, [i for i in live_idx if i in used]

VENDOR_B = [r"\bto ([A-Z][A-Za-z0-9&'.\- ]{1,50}?) previously", r"\bSupplier ([A-Z][A-Za-z0-9&'.\- ]{1,50}?)(?=;|,|\.|\s+invoice|\s+for\b)",
            r"\bvendor ([A-Z][A-Za-z0-9&'.\- ]{1,50}?)(?=;|,|\.|\s+for\b|\s+invoice)", r"\bfrom ([A-Z][A-Za-z0-9&'.\- ]{1,50}?)(?=;|,|\.|\s+for\b|\s+covers)"]
BAD_VENDOR = re.compile(r"^(The|This|A|An|Order|Payment|Vendor|Operations|Archive|Policy)\b|\bAC-", re.I)

def vendor_b(text):
    for rx in VENDOR_B:
        for m in re.finditer(rx, text):
            v = m.group(1).strip()
            if not BAD_VENDOR.search(v): return v
    return None

TEMPLATE = {
    "settle_invoice": "The records reconcile, no prior payment or bank-detail issue exists, and the total is within autonomous authority, so the invoice can be settled.",
    "request_approval": "The invoice is commercially valid and reconciled, but the total exceeds this queue's delegated payment authority, so a named approver must approve it.",
    "hold_invoice": "A required verification (such as an unverified bank-detail change or pending callback) is still outstanding, so payment must pause until it completes.",
    "reject_duplicate": "The same commercial invoice has already been paid, so duplicate-control policy requires rejection.",
    "open_exception": "Material signed records conflict beyond tolerance, so policy requires a documented exception case rather than settlement or hold.",
}

# ---------- AI layer ----------
SYSTEM = """You are a senior accounts-payable controller. For EACH invoice package choose exactly one action.

ACTIONS
- settle_invoice: invoice is valid, reconciled (PO/receipt/contract agree), no prior payment, no bank-change problem, and within autonomous payment authority.
- request_approval: commercially valid and reconciled, but the amount exceeds the delegation limit of this queue, so a named approver must approve.
- hold_invoice: payment must pause until a stated verification completes (e.g. an unverified bank-detail change / callback pending, tax or ID check, missing document).
- reject_duplicate: the SAME commercial invoice/order was already paid or discharged (same supplier, invoice identifier and total). A payment for a DIFFERENT invoice, order or amount is NOT a duplicate.
- open_exception: material records conflict (price, quantity, receipt, PO, contract, total outside tolerance) and policy requires a documented exception case.
Priority when several apply: reject_duplicate > open_exception > hold_invoice > request_approval > settle_invoice.

HOW TO READ
A package is a list of numbered paragraphs taken from several documents. Most are noise: boilerplate, cover sheets, archived or training examples, expired or superseded policies, closed incidents, notes about other suppliers or other orders, hypotheticals, and sentences that merely contain action words. NEVER decide by keywords.
Read the ENTIRE controlling paragraph. A positive opening ("clean three-way match", "otherwise payable", "no discrepancy remains") never overrides a later limit, duplicate, tolerance or verification statement. Compare the stated total with any authority/delegation limit numerically: total above the limit -> request_approval.
Two layouts occur:
 (a) One paragraph carries exactly three bracketed references such as [R_ABC123] and states the controlling current fact for THIS invoice.
 (b) Many short paragraphs each carry ONE reference. Then the decisive evidence is spread over several of them: the LIVE facts about this invoice/order plus the governing CURRENT payment-control policy. Archive, training, expired, superseded, closed-incident, unrelated-supplier and unrelated-order paragraphs are decoys.

"refs": return EXACTLY three different bracketed references, copied character-for-character INCLUDING the square brackets, from THIS package only. Layout (a): the three references of the controlling paragraph. Layout (b): the three most decisive tagged paragraphs (live facts plus the governing policy). Never include the cover-sheet reference or any decoy reference.
"paragraph": the number n of the main controlling paragraph.

FACTS
vendorName is the supplier name only, WITHOUT a leading word such as "Supplier". invoiceNumber exactly as written (e.g. INV-2024-267254). amountMinor is an integer in minor units (x100 for INR/USD/EUR/GBP; x1 for JPY/KRW). currency is the ISO code. If a fact is genuinely not stated, use "" or 0; never guess.

Answer for EVERY package index you are given. Output ONLY JSON:
{"decisions":[{"index":0,"paragraph":0,"refs":["[R_...]","[R_...]","[R_...]"],"action":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},"reason":"1-2 sentences naming the controlling fact"}]}"""

def to_int(a):
    try:
        if isinstance(a, bool): return 0
        if isinstance(a, int): return a
        if isinstance(a, float): return int(round(a))
        if isinstance(a, str): return int(re.sub(r"[^\d]", "", a) or 0)
    except Exception:
        pass
    return 0

def build(pk, d, allow_no_model=False):
    """Turn a (possibly empty/bad) model decision into a validated decision. Returns None to retry."""
    try:
        d = d if isinstance(d, dict) else {}
        A = analyze(pk)
        ps, text, refs_of = A["ps"], A["text"], A["refs_of"]
        model_action = d.get("action") if d.get("action") in ACTIONS else None
        if not model_action and d:
            found = [a for a in ACTIONS if a in json.dumps(d)]
            if len(found) == 1: model_action = found[0]
        if not model_action and not allow_no_model:
            print("build: no/invalid model action", str(d)[:200]); return None
        action = model_action

        # ----- facts -----
        mf = d.get("facts") if isinstance(d.get("facts"), dict) else {}
        cf = A["facts"]
        facts = {
            "vendorName": cf.get("vendorName") or re.sub(r"^(supplier|vendor)\s+", "", str(mf.get("vendorName", "")).strip(), flags=re.I).strip() or "unknown",
            "invoiceNumber": cf.get("invoiceNumber") or str(mf.get("invoiceNumber", "")).strip() or "unknown",
            "amountMinor": cf.get("amountMinor") or to_int(mf.get("amountMinor", 0)),
            "currency": cf.get("currency") or str(mf.get("currency", "")).strip().upper() or "INR",
        }
        total = facts["amountMinor"]

        if A["tri"]:
            # ----- layout (a): one paragraph with three refs controls -----
            try: n = int(d.get("paragraph"))
            except Exception: n = -1
            if n not in A["tri"]: n = A["tri"][0]
            refs = refs_of[n]
            hits = signals(ps[n], total)
            if len(hits) == 1:
                action = next(iter(hits))
            elif hits and action not in hits:
                action = top(hits)
            elif not hits and clean_count(ps[n]) >= 2:
                action = "settle_invoice"        # clearly reconciled, nothing blocking
            elif not action:
                action = "hold_invoice"          # no model answer and no signal: never settle blind
        else:
            # ----- layout (b): evidence spread over single-ref paragraphs -----
            tagged = [i for i, r in enumerate(refs_of) if len(r) == 1]
            live = [i for i in tagged if not DECOY.search(ps[i]) or re.search(r"Payment controls", ps[i], re.I)]
            live_text = "\n".join(ps[i] for i in live)
            s_action, info, used = solve_b(live_text, live, ps)
            if s_action:
                action = s_action                     # structured comparison wins
            elif not action:
                hits = set().union(*[signals(ps[i], total) for i in live if not re.search(r"Payment controls", ps[i], re.I)]) if live else set()
                action = top(hits) or ("settle_invoice" if clean_count(live_text) >= 2 else "hold_invoice")
            # facts for layout b
            if not cf.get("amountMinor") and info.get("total"):
                facts["amountMinor"], facts["currency"] = info["total"], info["cur"] or facts["currency"]
                total = facts["amountMinor"]
            if not cf.get("vendorName"):
                v = vendor_b(live_text) or vendor_b(text)
                mv = re.sub(r"^(supplier|vendor)\s+", "", str(mf.get("vendorName", "")).strip(), flags=re.I).strip()
                facts["vendorName"] = mv or v or "unknown"
            if facts["invoiceNumber"] == "unknown":
                m = INVNO.search(live_text) or INVNO.search(text)
                if m: facts["invoiceNumber"] = m.group(0)
            bad = {refs_of[i][0] for i in tagged if i not in live}
            mrefs = d.get("refs") if isinstance(d.get("refs"), list) else []
            mrefs = uniq(["[" + x.strip().strip("[]").strip() + "]" for x in mrefs if isinstance(x, str) and x.strip("[] ")])
            mrefs = [r for r in mrefs if r in text and r not in bad]
            struct_refs = [refs_of[i][0] for i in used]
            live_refs = [refs_of[i][0] for i in live]
            if len(live_refs) == 3:
                refs = live_refs
            elif action == model_action and len(mrefs) == 3:
                refs = mrefs
            else:
                refs = uniq(struct_refs + mrefs + live_refs)[:3]
            if len(refs) < 3:
                refs = uniq(refs + [r[0] for r in refs_of if r] + REF.findall(text))[:3]
        if len(refs) != 3:
            print("build: could not find 3 refs"); return None

        reason = str(d.get("reason", "")).strip() if action == model_action else ""
        if len(reason) < 20: reason = TEMPLATE[action]
        return {"action": action, "facts": facts, "evidenceRefs": refs, "reason": reason[:800]}
    except Exception as e:
        print("build error:", repr(e)[:200])
        return None

async def ask(items, ctx, temp=0):
    payload = {"batchContext": ctx,
               "packages": [{"index": i, "paragraphs": [{"n": n, "text": t} for n, t in enumerate(paras(p)) if keep(t)]}
                            for i, p in items]}
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    if not REASONING_MODEL:
        body["temperature"] = temp
        body["max_tokens"] = 4000
    if os.environ.get("REASONING_EFFORT"):
        body["reasoning_effort"] = os.environ["REASONING_EFFORT"]
    if os.environ.get("JSON_MODE", "1") == "1":
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        r = await c.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"}, json=body)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt).strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    res = {}
    for x in json.loads(txt).get("decisions", []):
        try: res[int(x["index"])] = x
        except Exception: pass
    return res

def pkey(pk):
    return CACHE_V + sha(canon({k: v for k, v in pk.items() if k != "packageId"}))

def fallback(pk):  # only when the model fails; deterministic, never settles blind, never cached
    d = build(pk, {}, allow_no_model=True)
    if d: return d
    refs = (uniq(REF.findall(flat(pk))) + ["[?1]", "[?2]", "[?3]"])[:3]
    return {"action": "hold_invoice",
            "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs, "reason": TEMPLATE["hold_invoice"]}

async def work(todo, pkgs, ctx, res):
    for attempt in range(3):
        if not todo: return
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]

        async def one(ch):
            try:
                return await ask([(i, pkgs[i]) for i in ch], ctx, 0 if attempt == 0 else 0.3)
            except Exception as e:
                print("LLM ERROR:", repr(e)[:400])
                return {}

        got = {}
        for r in await asyncio.gather(*[one(c) for c in chunks]):
            got.update(r)
        for i in list(todo):
            d = build(pkgs[i], got.get(i) or {})
            if d:
                res[i] = d; cache_put(pkey(pkgs[i]), d); todo.remove(i)
            else:
                print("REJECTED decision for package", i, str(got.get(i))[:300])

async def decide(pkgs, ctx):
    res, todo = [None] * len(pkgs), []
    for i, p in enumerate(pkgs):
        c = cache_get(pkey(p))
        if c: res[i] = c
        else: todo.append(i)
    if todo:
        task = asyncio.create_task(work(todo, pkgs, ctx, res))  # keeps running and caching even if we time out
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=DECIDE_BUDGET)
        except Exception as e:
            print("decide timeout/error:", repr(e)[:200])
    return [r if r else fallback(pkgs[i]) for i, r in enumerate(res)]

def make_proposal(pk, d):
    refs, act = d["evidenceRefs"], d["action"]
    rat = f"Action {act} selected. {str(d.get('reason', ''))[:900]} Decisive evidence: {refs[0]}, {refs[1]}, {refs[2]}."
    if len(rat) < 60:
        rat += " The cited references are the controlling facts for this invoice package."
    return {"packageId": pk["packageId"], "actionId": "act-" + uuid.uuid4().hex, "action": act,
            "facts": d["facts"], "evidenceRefs": refs, "rationale": rat[:1500]}

# ---------- app ----------
app = FastAPI()
router = APIRouter(prefix=BASE_PATH)

CARD = {
    "name": "Invoice Action Agent",
    "description": "Reads invoice packages, proposes evidence-backed business actions, and executes only grader-approved ones.",
    "version": "1.0.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
    "defaultInputModes": [BATCH_MT, RESULT_MT],
    "defaultOutputModes": [PROP_MT, REC_MT],
    "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice with cited evidence.",
                "tags": ["invoice", "accounts-payable", "approvals"]}],
}

@app.get("/selftest")
async def selftest():
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                             json={"model": MODEL, "messages": [{"role": "user", "content": "Say OK"}]})
        return {"ok": r.status_code == 200, "llm_status": r.status_code, "model": MODEL}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}

@app.get("/.well-known/agent-card.json")
async def card():
    return Response(json.dumps(CARD), media_type="application/json")

@router.post("/message:send")
async def send(req: Request):
    p, e = gate(req, True)
    if e: return e
    try: body = await req.json()
    except Exception: return err(400, "INVALID_ARGUMENT", "Invalid JSON")
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"]
            or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or not msg["parts"]
            or not isinstance(msg["parts"][0], dict)):
        return err(400, "INVALID_ARGUMENT", "Malformed message")
    h, mid = sha(canon(msg)), msg["messageId"]
    async with locks[("m", p, mid)]:
        row = q1("select hash, task_id from msgs where principal=? and mid=?", (p, mid))
        if row:
            if row[0] != h: return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
            return out(200, {"task": load_task(row[1], p)})
        part = msg["parts"][0]
        if part.get("mediaType") == BATCH_MT: return await start(p, msg, h, part.get("data"))
        if part.get("mediaType") == RESULT_MT: return await finish(p, msg, h, part.get("data"))
        return err(400, "INVALID_ARGUMENT", "Unsupported part")

async def start(p, msg, h, data):
    pk = data.get("packages") if isinstance(data, dict) else None
    if (not isinstance(pk, list) or not pk or not data.get("batchId")
            or not all(isinstance(x, dict) and isinstance(x.get("packageId"), str) for x in pk)
            or len({x["packageId"] for x in pk}) != len(pk)):
        return err(400, "INVALID_ARGUMENT", "Malformed batch")
    print("SAMPLE:", canon(pk[0])[:2500])
    ctx = {k: v for k, v in data.items() if k != "packages"}
    decisions = await decide(pk, ctx)
    for a, b in zip(pk, decisions):
        print("DECISION", a["packageId"], b["action"], b["evidenceRefs"], b["facts"])
    tid, cid = str(uuid.uuid4()), str(uuid.uuid4())
    m = dict(msg); m["taskId"], m["contextId"] = tid, cid
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid, "name": "invoice-action-proposals",
                           "parts": [{"mediaType": PROP_MT, "data": {"batchId": data["batchId"],
                                      "proposals": [make_proposal(a, b) for a, b in zip(pk, decisions)]}}]}],
            "history": [m]}
    save_task(task, p, (p, msg["messageId"], h, tid))
    return out(200, {"task": task})

async def finish(p, msg, h, data):
    tid, cid = msg.get("taskId"), msg.get("contextId")
    if not isinstance(tid, str): return err(400, "INVALID_ARGUMENT", "taskId required")
    async with locks[("t", tid)]:
        t = load_task(tid, p)
        if not t: return err(404, "NOT_FOUND", "Task not found")
        if cid != t["contextId"]: return err(400, "INVALID_ARGUMENT", "Context mismatch")
        if t["status"]["state"] in TERMINAL: return err(409, "TASK_TERMINAL", "Task is already terminal")
        stored = t["artifacts"][0]["parts"][0]["data"]
        results = data.get("results") if isinstance(data, dict) else None
        if (not isinstance(data, dict) or data.get("batchId") != stored["batchId"]
                or not isinstance(results, list) or not results):
            return err(400, "INVALID_ARGUMENT", "Batch mismatch")
        by_id = {x["packageId"]: x for x in stored["proposals"]}
        seen, execs = set(), []
        for r in results:
            sp = by_id.get(r.get("packageId")) if isinstance(r, dict) else None
            if (not sp or r["packageId"] in seen or r.get("actionId") != sp["actionId"] or r.get("action") != sp["action"]
                    or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_ARGUMENT", "Result does not match proposal")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]:
                    return err(400, "INVALID_ARGUMENT", "Missing receipt")
                execs.append({"packageId": sp["packageId"], "actionId": sp["actionId"], "action": sp["action"],
                              "receiptNonce": r["receiptNonce"], "facts": sp["facts"], "evidenceRefs": sp["evidenceRefs"]})
        m = dict(msg)
        t["artifacts"].append({"artifactId": "receipts-" + tid, "name": "invoice-action-receipts",
                               "parts": [{"mediaType": REC_MT, "data": {"batchId": stored["batchId"], "executions": execs}}]})
        t["history"].append(m)
        t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save_task(t, p, (p, msg["messageId"], h, tid))
        return out(200, {"task": t})

@router.get("/tasks/{tid}")
async def get_task(tid: str, req: Request):
    p, e = gate(req)
    if e: return e
    t = load_task(tid, p)
    return out(200, t) if t else err(404, "NOT_FOUND", "Task not found")

@router.get("/tasks")
async def list_tasks(req: Request):
    p, e = gate(req)
    if e: return e
    rows = qall("select data from tasks where principal=? order by created desc", (p,))
    return out(200, {"tasks": [json.loads(r[0]) for r in rows]})

@router.post("/tasks/{tid}:cancel")
async def cancel(tid: str, req: Request):
    p, e = gate(req)
    if e: return e
    async with locks[("t", tid)]:
        t = load_task(tid, p)
        if not t: return err(404, "NOT_FOUND", "Task not found")
        if t["status"]["state"] in TERMINAL: return err(409, "TASK_TERMINAL", "Task is already terminal")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(t, p)
        return out(200, t)

app.include_router(router)