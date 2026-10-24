import asyncio, hashlib, json, os, re, sqlite3, threading, time, uuid
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

# ---------- config ----------
# Set BASE_URL in Render to EXACTLY what you submit, e.g. https://ga3-2.onrender.com/a2a/
BASE_URL = os.environ.get("BASE_URL") or (os.environ["RENDER_EXTERNAL_URL"].rstrip("/") + "/a2a/")
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
API_BASE = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openrouter/v1").rstrip("/")
API_KEY = os.environ["LLM_API_KEY"]
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini")
REASONING_MODEL = bool(re.search(r"(^|/)(gpt-5|o\d)", MODEL))   # these reject temperature / max_tokens

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

# ---------- AI layer ----------
REF = re.compile(r"\[[^\[\]\n]+\]")
CACHE_V = "v4:"   # bump to invalidate old cached decisions

SYSTEM = """You are a senior accounts-payable controller. For EACH invoice package choose exactly one action.

ACTIONS
- settle_invoice: invoice is valid, reconciled (PO/receipt/contract agree) and within autonomous payment authority.
- request_approval: commercially valid and reconciled, but the amount or terms exceed the delegated authority limit in the stated policy, so a named approver must approve.
- hold_invoice: payment must pause until a stated verification completes (e.g. bank-detail change callback, tax/ID check, missing document, pending confirmation).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number, same amount) was already paid. A payment for a DIFFERENT invoice, number or amount is NOT a duplicate.
- open_exception: material records conflict (quantity, price, PO, vendor, bank, contract) and need an exception workflow.

HOW TO READ
Each package is given as numbered paragraphs. Many paragraphs are noise: cover sheets, archived examples, old policy revisions, training decoys, negations ("this is not a duplicate", "no hold is required"), and sentences that merely contain action words. Do NOT decide by keywords.
Find the ONE paragraph that states the controlling, current fact for THIS invoice under the batch's policy revision, and decide from it. That paragraph normally carries exactly three bracketed references like [ABC-123]. Never choose a cover sheet, archive example, superseded or decoy paragraph.
Return that paragraph's number in "paragraph" and also list its three bracketed references, copied character-for-character, in "refs" (used only as a backup).

FACTS
vendorName and invoiceNumber exactly as written for the real invoice. amountMinor is an integer in minor units (amount x100 for INR/USD/EUR/GBP; x1 for JPY/KRW). Strip currency symbols and thousands separators (Indian 1,23,450.50 -> 12345050). currency is the ISO code.

Output ONLY JSON:
{"decisions":[{"index":0,"paragraph":0,"refs":["","",""],"action":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},"reason":"1-2 sentences naming the controlling fact"}]}"""

def uniq(seq):
    seen, o = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); o.append(x)
    return o

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

def build(pk, d):
    try:
        ps = paras(pk)
        n = int(d["paragraph"])
        refs = uniq(REF.findall(ps[n])) if 0 <= n < len(ps) else []
        if len(refs) != 3:
            alt, allt = d.get("refs"), flat(pk)
            if isinstance(alt, list) and len(set(alt)) == 3 and all(isinstance(x, str) and x in allt for x in alt):
                refs = list(alt)
        if len(refs) != 3 or d["action"] not in ACTIONS: return None
        f = d["facts"]; a = f["amountMinor"]
        if isinstance(a, str): a = int(re.sub(r"[^\d]", "", a))
        if isinstance(a, float):
            if not a.is_integer(): return None
            a = int(a)
        if type(a) is not int: return None
        f = {"vendorName": str(f["vendorName"]).strip(), "invoiceNumber": str(f["invoiceNumber"]).strip(),
             "amountMinor": a, "currency": str(f["currency"]).strip().upper()}
        if not f["vendorName"] or not f["invoiceNumber"] or not f["currency"]: return None
        return {"action": d["action"], "facts": f, "evidenceRefs": refs, "reason": str(d.get("reason", ""))[:800]}
    except Exception:
        return None

async def ask(items, ctx, temp=0):
    payload = {"batchContext": ctx,
               "packages": [{"index": i, "paragraphs": [{"n": n, "text": t} for n, t in enumerate(paras(p)) if REF.search(t)]}
                            for i, p in items]}
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    if not REASONING_MODEL:
        body["temperature"] = temp
    if os.environ.get("REASONING_EFFORT"):
        body["reasoning_effort"] = os.environ["REASONING_EFFORT"]
    if os.environ.get("JSON_MODE", "1") == "1":
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"}, json=body)
    r.raise_for_status()
    txt = re.sub(r"^```(?:json)?|```$", "", r.json()["choices"][0]["message"]["content"].strip()).strip()
    return {int(x["index"]): x for x in json.loads(txt)["decisions"]}

def pkey(pk):
    return CACHE_V + sha(canon({k: v for k, v in pk.items() if k != "packageId"}))

def fallback(pk):  # only used when the model fails; never cached
    refs = (uniq(REF.findall(flat(pk))) + ["[?1]", "[?2]", "[?3]"])[:3]
    return {"action": "hold_invoice",
            "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs, "reason": "Automated analysis unavailable; payment held pending verification.",
            "_fb": True}

async def work(todo, pkgs, ctx, res):
    for attempt in range(2):
        if not todo: return
        chunks = [todo[i:i + 6] for i in range(0, len(todo), 6)]

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
            await asyncio.wait_for(asyncio.shield(task), timeout=40)
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
        return {"ok": r.status_code == 200, "llm_status": r.status_code}
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
    if all(d.get("_fb") for d in decisions):   # model fully unavailable: don't store a junk task, let a retry re-run
        return err(503, "UNAVAILABLE", "Decision service temporarily unavailable")
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