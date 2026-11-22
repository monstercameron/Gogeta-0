import sys, json
t = sys.stdin.read()
i = t.rfind("SELFTEST")
try:
    j = json.loads(t[:i])
    print(json.dumps(j["per_kind"]))
    print(j["minhash_abs_error"], j["families"], j["presence"], j["stream"]["ok"], j["substring"])
    print(t[i:])
except Exception as e:
    print(e, t[-3000:])
