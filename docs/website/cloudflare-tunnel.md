# Cloudflare Tunnel Deployment

Publishes MARINA at `https://marina.example.com` from a machine that has **no
public IP, no open inbound ports, and no port-forwarding**. A `cloudflared`
container opens an outbound connection to Cloudflare's edge; visitors reach
Cloudflare, and Cloudflare hands the request back down that connection.

This is the right path for a lab workstation or a GPU box behind NAT. TLS is
terminated at Cloudflare's edge, so you get HTTPS without managing certificates.

**Prerequisites:**

- Steps 1 and 2 of the [deployment guide](README.md) — model download and `.env`.
- A domain whose nameservers point at Cloudflare (free plan is sufficient).
  Adding a domain is done once in the Cloudflare dashboard under
  **Websites → Add a site**; it takes a few minutes to propagate.
- Docker Compose **v2.24+** — the overlay uses the `!reset` YAML tag.
  Check with `docker compose version`.

---

## What changes versus local deployment

The overlay file `docker-compose.cloudflared.yml` makes three changes:

| | Local | Tunnel |
|---|---|---|
| nginx host port | published (`NGINX_PORT`) | **none** — `ports: !reset []` |
| Route in | direct TCP to the host | outbound-only connection from `cloudflared` |
| Client IP source | `$remote_addr` | `CF-Connecting-IP`, via `nginx/realip-cloudflare.conf` |

`cloudflared` joins only `proxy-net`, so it can reach nginx and nothing else.
The backend stays on `backend-net`, two hops away from the tunnel.

---

## Step 1: Create the tunnel

In the [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com):

1. **Networks → Tunnels → Create a tunnel**
2. Choose **Cloudflared** as the connector type.
3. Name it (e.g. `marina`) and save.
4. On the install screen, ignore the suggested install commands — you only need
   the **token**. It is the long string after `--token` in the displayed command,
   starting with `ey…`. Copy it.

The token authenticates *and* identifies the tunnel; treat it as a credential.

---

## Step 2: Add the token to `.env`

```dotenv
MODEL_DATA_DIR=./checkpoints
DEVICE=cpu

# Tunnel credential — do not commit.
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoi...
```

`.env` is already gitignored. Verify before you commit anything:

```bash
git check-ignore -v .env && echo "ignored"
```

`NGINX_PORT` is irrelevant in this mode — the overlay removes the port mapping
entirely.

---

## Step 3: Configure the public hostname

Back in **Networks → Tunnels → marina → Configure → Published application routes**,
add a route:

| Field | Value |
|---|---|
| Subdomain | `marina` |
| Domain | `example.com` (your Cloudflare-managed domain) |
| Type | `HTTP` |
| URL | `nginx:80` |

`nginx:80` is the Compose service name on `proxy-net` — `cloudflared` resolves
it through Docker's internal DNS. **Not** `localhost:80`: cloudflared runs in
its own container, where `localhost` is itself.

Type is `HTTP`, not `HTTPS`. The hop from cloudflared to nginx is inside a
Docker bridge network; the public-facing TLS is handled by Cloudflare's edge.

Cloudflare creates the proxied DNS record for `marina.example.com` automatically.

---

## Step 4: Start the stack

```bash
cd /path/to/MARINA
docker compose -f docker-compose.yml -f docker-compose.cloudflared.yml up -d --build
```

That two-file invocation is required for **every** Compose command in this mode.
A plain `docker compose up -d` would apply only the base file — republishing the
host port and dropping `cloudflared`. Save yourself the mistake:

```bash
echo 'COMPOSE_FILE=docker-compose.yml:docker-compose.cloudflared.yml' >> .env
```

With that set, plain `docker compose up -d` / `logs` / `down` pick up both files.
The rest of this guide writes the flags out explicitly.

---

## Step 5: Verify

Watch the connector register — you want four `Registered tunnel connection`
lines (Cloudflare establishes redundant connections to different edge colos):

```bash
docker compose -f docker-compose.yml -f docker-compose.cloudflared.yml logs -f cloudflared
```

```
INF Connection <uuid> registered connIndex=0 location=lax01
INF Connection <uuid> registered connIndex=1 location=sjc07
```

Confirm nothing is listening on the host:

```bash
docker compose -f docker-compose.yml -f docker-compose.cloudflared.yml ps
# nginx's PORTS column should show only 80/tcp — no host mapping
```

Then check the public URL end to end:

```bash
curl -sf https://marina.example.com/api/health/live && echo " live"
curl -s  https://marina.example.com/api/health | python -m json.tool

curl -s -X POST https://marina.example.com/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"raw": {"hsqc": [7.23, 128.5, 1.0, 3.81, 55.2, 0.8], "mw": 350.0}, "k": 5}' \
  | python -m json.tool | head -20
```

---

## Step 6: Verify client IPs

This is the step most tunnel setups skip, and it silently breaks abuse control.

The edge nginx sets `X-Forwarded-For` to `$remote_addr` — deliberately
overwriting rather than appending, so callers cannot forge the address the
backend rate-limits on. Behind a tunnel, `$remote_addr` is the **cloudflared
container's IP**, identical for every visitor. Without a fix, all traffic shares
one rate-limit bucket (`30 per minute` for `/api/predict`, total, across the
whole internet) and `/api/stats` reports one unique visitor forever.

`nginx/realip-cloudflare.conf`, mounted by the overlay, restores the real
address from the `CF-Connecting-IP` header that Cloudflare's edge injects.

Confirm it is active:

```bash
docker compose -f docker-compose.yml -f docker-compose.cloudflared.yml \
  exec nginx ls /etc/nginx/conf.d/
# expect: 00-realip.conf  marina-proxy.conf
```

Then load the site from two different networks (e.g. laptop and phone on
cellular) and check that the visitor count increments:

```bash
curl -s https://marina.example.com/api/stats | python -m json.tool
```

`unique_clients` should rise by two, not one.

> **Do not mount `nginx/realip-cloudflare.conf` on a deployment that also
> publishes a host port.** It trusts `CF-Connecting-IP` from any source, so
> anyone able to reach nginx directly could set that header themselves and evade
> the rate limiter. It is safe here only because tunnel mode publishes no port.

---

## Timeouts

Cloudflare's edge returns **HTTP 524** if the origin takes longer than **100
seconds** to respond (free and Pro plans; not configurable below Enterprise).
The nginx `proxy_read_timeout` of 300s is therefore not the binding constraint.

The backend's own `PREDICT_TIMEOUT_S` defaults to **60s**, safely under the
limit — leave it there. If you raise it for large-`k` requests on slow CPU
inference, keep it below ~90s or clients will see a Cloudflare error page
instead of the backend's own 504.

---

## Locking the site down

The tunnel gives you HTTPS and hides your IP, but the site is still public. Two
things worth adding, both configured in the Cloudflare dashboard rather than in
this repo:

**Restrict access to specific people.** Zero Trust → **Access → Applications**
→ add a self-hosted application for `marina.example.com` with an email or
identity-provider policy. Visitors then authenticate at Cloudflare's edge before
any request reaches your machine. This is the right tool for an internal or
pre-publication deployment.

**Keep the API docs private.** Leave `EXPOSE_API_VIA_NGINX` at its default of
`false` and `/docs`, `/redoc`, and `/openapi.json` are never routed to the
backend. Note this hides the documentation, not the API: `/api/*` stays
reachable either way, because the SPA calls it from the browser. Access policies
are the only way to close that off — see
[the API surface](README.md#the-api-surface-and-what-expose_api_via_nginx-does).

**Rate limiting at the edge.** Cloudflare's own rate-limiting rules run before
traffic reaches your machine, which is strictly better than the in-process
limiter for absorbing floods. The backend limiter remains a useful backstop —
it is per worker process, so with `UVICORN_WORKERS > 1` the effective limit is
the configured value times the worker count.

---

## Common operations

Every command needs both files (or `COMPOSE_FILE` set, per step 4):

```bash
CF="-f docker-compose.yml -f docker-compose.cloudflared.yml"

docker compose $CF up -d --build       # rebuild + start
docker compose $CF logs -f cloudflared # tunnel connector logs
docker compose $CF logs -f backend     # inference logs
docker compose $CF ps                  # status
docker compose $CF restart nginx       # reload proxy config
docker compose $CF down                # stop everything
```

`scripts/website/start.sh` does **not** know about the overlay — it always
invokes the base file alone, which would republish the host port and drop the
tunnel. Use raw `docker compose` in this mode.

To switch back to local-only, bring the stack down with the overlay, then up
without it:

```bash
docker compose $CF down
docker compose up -d
```

---

## Troubleshooting

**`required variable CLOUDFLARE_TUNNEL_TOKEN is missing a value` at `up` time.**
The variable is missing from `.env`, or you are running Compose from a directory
other than the repo root. The overlay declares it required (`:?`) rather than
letting cloudflared start and fail obscurely.

**Tunnel connects, but the hostname returns Cloudflare error 1033 / 530.**
The tunnel is running with no published-application route, or the route points
at a hostname Cloudflare does not manage. Re-check step 3.

**Error 502 from Cloudflare.**
cloudflared reached the edge but cannot reach the origin. Almost always the
service URL: it must be `http://nginx:80`, not `localhost`. Verify from inside
the connector:

```bash
docker compose $CF exec cloudflared wget -qO- http://nginx:80/api/health/live
```

**Error 524 on predictions.**
Inference exceeded Cloudflare's 100-second origin timeout. Lower `k`, move to
GPU inference, or reduce `PREDICT_TIMEOUT_S` so the backend returns its own 504
first. See [Timeouts](#timeouts).

**Everyone gets 429 at once.**
The real-IP fix is not applied — all visitors are sharing one rate-limit bucket.
See [step 6](#step-6-verify-client-ips).

**Site loads, predictions fail.**
Not a tunnel problem — check `docker compose $CF logs backend` and see
[Troubleshooting](README.md#troubleshooting).

**Changes to `.env` seem ignored.**
`docker compose up -d` alone does not recreate containers whose config it
considers unchanged. Force it: `docker compose $CF up -d --force-recreate`.

---

## Alternative: locally-managed tunnel config

The dashboard-managed tunnel above stores routing in Cloudflare. If you would
rather keep the tunnel config in the repo — for reproducibility, or to route
several hostnames — use a credentials file instead of a token.

Create the tunnel with the `cloudflared` CLI on the host (`cloudflared tunnel
login`, then `cloudflared tunnel create marina`), which writes
`~/.cloudflared/<TUNNEL_ID>.json`. Then replace the `cloudflared` service in
`docker-compose.cloudflared.yml` with:

```yaml
  cloudflared:
    image: cloudflare/cloudflared:latest
    container_name: marina-cloudflared
    command: tunnel --no-autoupdate --config /etc/cloudflared/config.yml run
    volumes:
      - ~/.cloudflared:/etc/cloudflared:ro
    depends_on:
      - nginx
    networks:
      - proxy-net
    restart: unless-stopped
```

with `~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /etc/cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: marina.example.com
    service: http://nginx:80
  - service: http_status:404
```

Then point DNS at the tunnel once:

```bash
cloudflared tunnel route dns marina marina.example.com
```

Everything else in this guide — the removed host port, the real-IP fix, the
100-second timeout, the operational commands — applies unchanged. The
dashboard-managed token remains the simpler choice for a single hostname.
