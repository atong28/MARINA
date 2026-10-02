# Local Deployment

Runs the full stack with the edge nginx published on a host port. Use this for
`http://localhost`, for a lab machine reachable by IP, or as the first step
before adding a [Cloudflare Tunnel](cloudflare-tunnel.md).

**Prerequisite:** complete steps 1 and 2 of the [deployment guide](README.md) —
model download and `.env` — before starting here.

---

## Step 1: Confirm `.env`

```dotenv
MODEL_DATA_DIR=./checkpoints
NGINX_PORT=8080               # 80 needs root on Linux; 8080 avoids that
DEVICE=cpu
```

Add `EXPOSE_API_VIA_NGINX=true` if you also want the interactive API docs at
`/docs` — see [the API surface](README.md#the-api-surface-and-what-expose_api_via_nginx-does).
The site works either way.

Port 80 is the default. On Linux, binding it requires Docker to run as root
(the usual setup) but will collide with any system web server; `8080` is the
easier choice for a dev machine.

---

## Step 2: Build and start

```bash
cd /path/to/MARINA
docker compose build
docker compose up -d
```

The backend image compiles PyTorch wheels and is slow to build the first time
(10–20 minutes is normal). Subsequent builds are cached.

Watch the backend come up — it loads the model before serving traffic, which
takes another 1–2 minutes:

```bash
docker compose logs -f backend
```

Wait for:

```
Model marina_db_s1 ready
MARINA backend ready
```

The Compose healthcheck has a 90-second `start_period` for exactly this reason;
`docker compose ps` will read `starting` until the model is loaded.

---

## Step 3: Verify

```bash
curl -sf http://localhost:8080/api/health/live && echo " live"
curl -s  http://localhost:8080/api/health | python -m json.tool
```

Then open <http://localhost:8080> in a browser and run one of the built-in
example spectra. If anything misbehaves, see
[Troubleshooting](README.md#troubleshooting).

---

## Step 4 (optional): GPU inference

Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
then uncomment the reservation block on the `backend` service in
`docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Set the CUDA wheel index and device in `.env`:

```dotenv
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128   # or .../cu124
DEVICE=cuda
```

Rebuild and restart:

```bash
docker compose build backend
docker compose up -d --force-recreate
```

Confirm the GPU is visible inside the container:

```bash
docker compose exec backend python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Exposing on a LAN

Docker's published port binds `0.0.0.0` by default, so the site is already
reachable at `http://<host-ip>:$NGINX_PORT` from the same network — subject to
the host firewall:

```bash
sudo ufw allow 8080/tcp          # Ubuntu/Debian
```

Before doing this, understand what you are publishing:

- **There is no HTTPS and no authentication.** Everything, including spectral
  data submitted by users, crosses the network in plaintext. For anything
  beyond a trusted lab network, use the [Cloudflare Tunnel](cloudflare-tunnel.md)
  path, which terminates TLS at Cloudflare's edge.
- **The inference API is public** to anyone who can reach the port, and cannot
  be hidden by configuration — the SPA calls it from the browser. With
  `EXPOSE_API_VIA_NGINX=true` the docs at `/docs` and `/openapi.json` are public
  too. Use authentication if that is not acceptable.
- **Rate limits are per worker process, not global.** With
  `UVICORN_WORKERS > 1` the effective limit is the configured value times the
  worker count.
- The backend rate-limits on `X-Forwarded-For`, which the edge nginx overwrites
  with the real `$remote_addr`. On a directly published port this is correct
  with no extra configuration. Do **not** mount `nginx/realip-cloudflare.conf`
  here — it would let any client spoof its own IP via a `CF-Connecting-IP`
  header and bypass the limiter.

To restrict the site to the host machine only, change the port mapping in
`docker-compose.yml` to bind loopback:

```yaml
    ports:
      - "127.0.0.1:${NGINX_PORT:-8643}:80"
```

---

## Stopping and cleaning up

```bash
docker compose down                 # stop and remove containers
docker compose down -v              # also delete the marina-state volume
                                    # (resets the usage/visitor counters)
```

Model files in `MODEL_DATA_DIR` live on the host and are never touched — the
mount is read-only.

---

## Next steps

- Give the deployment a public HTTPS hostname → [Cloudflare Tunnel](cloudflare-tunnel.md)
- Tune backend behaviour → [`backend/README.md`](../../backend/README.md#environment-variables-reference)
- Add a second model or a SPECTRE checkpoint → [`backend/README.md`](../../backend/README.md#multi-model-setup)
