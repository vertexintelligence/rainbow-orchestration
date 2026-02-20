#!/usr/bin/env python3
import argparse, os, json, time, uuid, hashlib, hmac, subprocess, sys

ENV_PATH_DEFAULT = os.path.expanduser("~/genx/secrets/broker.env")
BROKER_URL_DEFAULT = "http://localhost:8787"

def load_env(path: str) -> dict:
    vals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k] = v
    return vals

def sign_request(actor: str, path: str, body_bytes: bytes, env_vals: dict):
    actor_keys = json.loads(env_vals.get("ACTOR_KEYS_JSON", "{}"))
    hmac_keys  = json.loads(env_vals.get("HMAC_KEYS_JSON", "{}"))

    if actor not in actor_keys:
        raise SystemExit("ERROR: actor missing in ACTOR_KEYS_JSON")

    kid = actor_keys[actor]
    if kid not in hmac_keys:
        raise SystemExit("ERROR: key-id missing in HMAC_KEYS_JSON")

    secret = hmac_keys[kid].encode("utf-8")
    ts = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    canonical = f"POST\n{path}\n{ts}\n{nonce}\n{body_sha}".encode("utf-8")
    sig = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
    return kid, ts, nonce, sig

def curl_post(url: str, path: str, headers: dict, body_bytes: bytes) -> str:
    cmd = ["curl", "-sS", "-X", "POST", url + path, "-H", "content-type: application/json"]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    cmd += ["--data-binary", body_bytes.decode("utf-8")]
    return subprocess.check_output(cmd).decode("utf-8", errors="replace")

def main():
    ap = argparse.ArgumentParser(description="GenX signed client for broker endpoints")
    ap.add_argument("--actor", required=True)
    ap.add_argument("--env", default=ENV_PATH_DEFAULT)
    ap.add_argument("--url", default=BROKER_URL_DEFAULT)
    ap.add_argument("--thread", default="t_default")
    ap.add_argument("--model", default="mistral:latest")
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.2)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chat")
    g.add_argument("--code")

    args = ap.parse_args()
    env_vals = load_env(args.env)

    if args.chat is not None:
        path = "/v1/chat"
        body = {
            "actor": args.actor,
            "intent": "chat",
            "payload": {
                "thread_id": args.thread,
                "messages": [{"role": "user", "content": args.chat}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "model": args.model,
            },
        }
    else:
        path = "/v1/route"
        body = {
            "actor": args.actor,
            "intent": "run_code",
            "payload": {"language": "python", "code": args.code},
        }

    body_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    kid, ts, nonce, sig = sign_request(args.actor, path, body_bytes, env_vals)

    headers = {
        "x-genx-actor": args.actor,
        "x-genx-key-id": kid,
        "x-genx-ts": ts,
        "x-genx-nonce": nonce,
        "x-genx-sig": sig,
    }

    print(curl_post(args.url, path, headers, body_bytes))

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.output.decode("utf-8", errors="replace") if e.output else str(e))
        sys.exit(e.returncode)
