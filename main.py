import asyncio, hashlib, json, os, re, sqlite3, threading, time, uuid
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

BASE_URL = os.environ.get("BASE_URL") or (os.environ["RENDER_EXTERNAL_URL"].rstrip("/") + "/a2a/")
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
API_BASE = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ["LLM_API_KEY"]
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

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
    if isinstance(o, dict): return "\n".join(flat(v) for v in o.values())
    if isinstance(o, list): return "\n".join(flat(v) for v in o)
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
SYSTEM = """You are an accounts-payable analyst. For EACH package decide exactly one action:
- settle_invoice: valid, reconciled, within autonomous approval authority.
- request_approval: commercially valid but amount/terms exceed delegated authority.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number, amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (PO/receipt/contract/bank mismatch) and need an exception workflow.
Rules: documents contain old examples, archive samples, training decoys, negations ("not a duplicate", "no hold required") and irrelevant action words. Ignore those. Find the ONE paragraph that actually determines the action for THIS invoice under the stated policy. Return exactly the THREE bracketed references from that paragraph, copied character-for-character as written in the documents. Do NOT include the cover-sheet reference, archive examples or decoys.
facts.amountMinor is an integer in minor units (INR/USD/EUR x100; JPY x1).
Respond ONLY with JSON: {"decisions":[{"index":0,"action":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},"evidenceRefs":["","",""],"reason":"one or two sentences"}]}"""

def norm(d):
    try:
        f = d["facts"]; a = f["amountMinor"]
        if isinstance(a, float) and a.is_integer(): f["amountMinor"] = int(a)
        elif isinstance(a, str) and re.fullmatch(r"\d+", a): f["amountMinor"] = int(a)
        return d
    except Exception:
        return None

def valid(d, text):
    try:
        f, refs = d["facts"], d["evidenceRefs"]
        return (d["action"] in ACTIONS and isinstance(refs, list) and len(refs) == 3 and len(set(refs)) == 3
                and all(isinstance(r, str) and r in text for r in refs)
                and isinstance(f["vendorName"], str) and isinstance(f["invoiceNumber"], str)
                and type(f["amountMinor"]) is int and isinstance(f["currency"], str))
    except Exception:
        return False

async def ask(items):
    body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps([{"index": i, "package": p} for i, p in items], ensure_ascii=False)}]}
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"}, json=body)
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    return {int(d["index"]): d for d in data["decisions"]}

def pkey(pk):
    return sha(canon({k: v for k, v in pk.items() if k != "packageId"}))

def fallback(pk):  # never settles when the model fails; not cached
    refs = (re.findall(r"\[[^\]]+\]", flat(pk)) + ["[?1]", "[?2]", "[?3]"])[:3]
    return {"action": "hold_invoice", "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs, "reason": "Automated analysis unavailable; payment held pending manual verification."}

async def decide(pkgs):
    res, todo = [None] * len(pkgs), []
    for i, p in enumerate(pkgs):
        c = cache_get(pkey(p))
        if c: res[i] = c
        else: todo.append(i)
    for _ in range(2):
        if not todo: break
        try: got = await ask([(i, pkgs[i]) for i in todo])
        except Exception: got = {}
        for i in list(todo):
            d = norm(got.get(i))
            if d and valid(d, flat(pkgs[i])):
                res[i] = d; cache_put(pkey(pkgs[i]), d); todo.remove(i)
    for i in todo: res[i] = fallback(pkgs[i])
    return res

def make_proposal(pk, d):
    refs, act = d["evidenceRefs"], d["action"]
    rat = f"Action {act} selected. {str(d.get('reason',''))[:900]} Decisive evidence: {refs[0]}, {refs[1]}, {refs[2]}."
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
    if not isinstance(pk, list) or not pk or not data.get("batchId") or not all(isinstance(x, dict) and isinstance(x.get("packageId"), str) for x in pk) \
            or len({x["packageId"] for x in pk}) != len(pk):
        return err(400, "INVALID_ARGUMENT", "Malformed batch")
    decisions = await decide(pk)
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
        if not isinstance(data, dict) or data.get("batchId") != stored["batchId"] or not isinstance(results, list) or not results:
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