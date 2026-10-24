import httpx, uuid, os
B = os.environ["BASE_URL"].rstrip("/")
H = lambda t="tokA": {"Authorization": f"Bearer {t}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
O = B.split("/a2a")[0]
print("card", httpx.get(O + "/.well-known/agent-card.json").status_code)
print("noauth", httpx.get(B + "/tasks").status_code)
print("badver", httpx.get(B + "/tasks", headers={**H(), "A2A-Version": "0.3"}).status_code)

def msg(mid, text):
    return {"message": {"messageId": mid, "role": "ROLE_USER", "parts": [{
        "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
        "data": {"batchId": "b1", "policyRevision": "r1", "packages": [{"packageId": "p1", "text": text}]}}]}}

mid = str(uuid.uuid4())
sample = "Vendor Acme, invoice INV-1, INR 500.00. [DOC-1] [DOC-2] [DOC-3] Amount within limit, PO matched."
r1 = httpx.post(B + "/message:send", headers=H(), json=msg(mid, sample), timeout=60).json()["task"]
print(r1["status"]["state"], r1["artifacts"][0]["parts"][0]["data"]["proposals"][0]["action"])
r2 = httpx.post(B + "/message:send", headers=H(), json=msg(mid, sample), timeout=60).json()["task"]
print("same task on replay:", r1["id"] == r2["id"])
print("conflict:", httpx.post(B + "/message:send", headers=H(), json=msg(mid, "changed")).status_code)
print("outsider get:", httpx.get(f"{B}/tasks/{r1['id']}", headers=H("tokB")).status_code,
      "outsider list:", httpx.get(B + "/tasks", headers=H("tokB")).json())
prop = r1["artifacts"][0]["parts"][0]["data"]["proposals"][0]
res = {"message": {"messageId": str(uuid.uuid4()), "taskId": r1["id"], "contextId": r1["contextId"], "role": "ROLE_USER",
       "parts": [{"mediaType": "application/vnd.ga5.invoice-action-results+json", "data": {"batchId": "b1", "results": [
           {"packageId": "p1", "actionId": prop["actionId"], "action": prop["action"], "outcome": "ACCEPTED", "receiptNonce": "n-123"}]}}]}}
t = httpx.post(B + "/message:send", headers=H(), json=res).json()["task"]
print(t["status"]["state"], len(t["artifacts"]))
print("cancel after done:", httpx.post(f"{B}/tasks/{t['id']}:cancel", headers=H()).status_code)  # expect 409