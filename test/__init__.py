import redis

r = redis.Redis(host='localhost', port=32768, decode_responses=True)

r.set("test", "123")
print(r.get("test"))

r.delete("test")

print(r.keys("*"))