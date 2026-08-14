# Deploying on k3s (Jetson)

Run the whole Acoustic Analyzer stack as Kubernetes pods on the Jetson with **k3s** — everything
self-contained, with **PVCs backed by the NVMe** and **Ollama on the GPU**.

Ships three pods (namespace `acoustic`):

| Pod | Image | Notes |
|-----|-------|-------|
| **ollama** | `ollama/ollama` | the LLM · GPU · models PVC on NVMe |
| **acoustic-tools** | built locally | the DSP tool service — NodePort `:30800` (`/analyze`, `/spectrum`) |
| **acoustic-chat** | reuses the built image | reusable agent engine (`src/agent_shell.py`) + this vertical's prompt, served as a **ttyd web terminal** — NodePort `:30088` |

The front-end is a **terminal UI**, not a web app: open `http://<jetson>:30088` in a browser and
you get the agent's REPL. Manifests: [`deploy/k8s/`](../../deploy/k8s/).

> **Why k3s here:** lightweight, official arm64, ideal for a single edge node. It uses
> **containerd**, not Docker — which changes how the local image gets in (step 4).

> 🏷️ **Hostnames in this doc are placeholders.** Every example uses **`jetson.local`**. Either name
> your board `jetson` (`sudo hostnamectl set-hostname jetson`, see §"Give the Jetson a home DNS
> name") so that resolves via mDNS, or substitute your own hostname throughout — including in the
> registry address, the manifests' `image:` lines, and `PUBLIC_URL`. The `Makefile` reads
> `REG`, so `make deploy REG=<your-host>:30500` overrides it without editing anything; put it in a
> gitignored `Makefile.local` to make it permanent.

---

## 1 · Install k3s (with TLS names + a readable kubeconfig)

Set the API-cert names and kubeconfig permissions **before** installing — then the cert is valid
for your hostname/IP from the first boot, with no cert regeneration later. k3s reads
`/etc/rancher/k3s/config.yaml` at startup (you create it):

First make sure the NVMe is mounted and persistent (so k3s can live on it):
```bash
lsblk -o NAME,SIZE,MOUNTPOINT | grep nvme      # e.g. nvme0n1p1 -> /mnt/nvme
grep -q /mnt/nvme /etc/fstab || echo "!! add the NVMe to /etc/fstab so it mounts at boot"
```

```bash
sudo mkdir -p /etc/rancher/k3s
sudo tee /etc/rancher/k3s/config.yaml >/dev/null <<'EOF'
data-dir: /mnt/nvme/k3s                # store ALL k3s data (images + PVCs) on the fast NVMe
write-kubeconfig-mode: "0644"          # your user can read the kubeconfig (no sudo/chown needed)
tls-san:                               # names/IPs the API cert must be valid for
  - jetson.local
  - jetson.lan
  - 10.0.0.200
disable:
  - traefik                            # we use NodePort, not Ingress — skips Traefik + its svclb pod
EOF

curl -sfL https://get.k3s.io | sh -

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml    # add to ~/.bashrc
kubectl get nodes                               # Ready = good
```

> `write-kubeconfig-mode: 0644` makes the admin kubeconfig readable by any local user — fine on a
> single-user Jetson. Prefer stricter? Drop that line and instead copy it to your user:
> `sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $USER ~/.kube/config`.
>
> **Already installed** without the config file? Create it now with the same contents, then
> regenerate the cert: `sudo rm -f /var/lib/rancher/k3s/server/tls/dynamic-cert.json && sudo systemctl restart k3s` (details in §9).

---

## 2 · GPU for Ollama (required)

Ollama needs the Jetson GPU. JetPack installs `nvidia-container-runtime`; k3s auto-wires it into
containerd **at install time**. Verify it's there:

```bash
sudo grep -i nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml && echo "nvidia runtime present"
```

- **Present** → good; the `nvidia` RuntimeClass is created by `ollama.yaml`.
- **Missing** (k3s was installed before the runtime) → reinstall/restart k3s so it regenerates the
  config: `curl -sfL https://get.k3s.io | sh -` again, then `sudo systemctl restart k3s`.

The Ollama pod selects the GPU with `runtimeClassName: nvidia` + `NVIDIA_VISIBLE_DEVICES=all`
(already in the manifest). No device-plugin needed — the Jetson iGPU is shared, not a schedulable
discrete GPU.

> If Ollama ends up on CPU (slow), switch the image in [`ollama.yaml`](../../deploy/k8s/ollama.yaml)
> to **`dustynv/ollama:latest`** — a Jetson-optimized build — and re-apply.

---

## 3 · Put k3s data (images + PVCs) on the NVMe

k3s keeps **everything** under `/var/lib/rancher/k3s` — container **images** at
`.../agent/containerd` and local-path **PVCs** at `.../storage`. On a stock Jetson the root
filesystem (and thus this directory) is on the **slow SD card**, which will bottleneck image
pulls, container I/O, and the ollama model store. Check where it lives:

```bash
df -h /var/lib/rancher/k3s        # /dev/mmcblk0p1 = SD card;  /dev/nvme0n1p1 = NVMe
lsblk -o NAME,SIZE,MOUNTPOINT     # where is the NVMe mounted? (e.g. /mnt/nvme)
```

If you set `data-dir: /mnt/nvme/k3s` at install (§1), it's **already on the NVMe** — done.

Otherwise (installed on SD already, don't want to reinstall), relocate the whole k3s data dir to
the NVMe and **bind-mount** it back — one move fixes **both** images and PVCs, and `rsync`
preserves the CA/certs (kubeconfig stays valid) plus any images already pulled (no re-pull, no reboot):

```bash
sudo systemctl stop k3s
sudo mkdir -p /mnt/nvme/k3s
sudo rsync -aHAX /var/lib/rancher/k3s/ /mnt/nvme/k3s/       # images + PVCs + certs
sudo mv /var/lib/rancher/k3s /var/lib/rancher/k3s.bak
sudo mkdir -p /var/lib/rancher/k3s
echo '/mnt/nvme/k3s  /var/lib/rancher/k3s  none  bind  0 0' | sudo tee -a /etc/fstab
sudo mount -a
sudo systemctl start k3s

df -h /var/lib/rancher/k3s        # now /dev/nvme0n1p1 ✓    then: sudo rm -rf /var/lib/rancher/k3s.bak
```
Ensure the NVMe mount `/mnt/nvme` itself is in `/etc/fstab` so it mounts before the bind at boot.

The `ollama-models` PVC uses the default `local-path` StorageClass, so once the data dir is on the
NVMe it lands there automatically.

> **Fuller fix:** move the *entire OS* to the NVMe with **rootOnNVMe**
> ([JETSON-SETUP.md](JETSON-SETUP.md)) — then root, k3s, images, and PVCs are all on the NVMe and
> the SD only holds the boot partition. The bind-mount above handles all the k3s/container I/O
> without a reboot; rootOnNVMe additionally moves the OS itself.

---

## 4 · Image workflow — an in-cluster registry (no sudo in the loop)

The everyday loop is `build → push → restart`, all via `kubectl`/`docker`, **no sudo**. It's
powered by a private registry running *in the cluster* on the NVMe
([`deploy/registry/registry.yaml`](../../deploy/registry/registry.yaml) — its own `registry` namespace — pushed/pulled at `jetson.local:30500`).
Your Mac and the Jetson are both **arm64**, so a Mac build runs on the Jetson as-is.

### One-time setup
1. **Start the registry:** `kubectl apply -f deploy/registry/registry.yaml` → check
   `curl http://jetson.local:30500/v2/` returns `200`.
2. **Tell k3s to trust it** (insecure HTTP on the LAN) — *the only sudo, once*:
   ```bash
   # on the Jetson:
   sudo tee /etc/rancher/k3s/registries.yaml >/dev/null <<'EOF'
   mirrors:
     "jetson.local:30500":
       endpoint:
         - "http://jetson.local:30500"
   EOF
   sudo systemctl restart k3s
   ```
3. **Let Docker push over HTTP** — on the Mac, Docker Desktop → Settings → Docker Engine, add:
   `"insecure-registries": ["jetson.local:30500"]` → Apply & Restart.

### The everyday loop (sudo-free, scriptable — see the [`Makefile`](../../Makefile))
```bash
make deploy       # docker build → docker push → kubectl rollout restart (tools + chat)
make restart      # just re-pull + restart (no rebuild)
make configmaps   # refresh the agent-shell / prompt ConfigMaps (no image)
make logs
```
The app manifests use `image: jetson.local:30500/acoustic-tools:latest` +
`imagePullPolicy: Always`, so each `rollout restart` pulls the freshly-pushed image.

> **Offline fallback** (no registry, e.g. first bring-up before setup): build + import a tarball —
> `docker save acoustic-tools:local | gzip | ssh <user>@jetson.local 'gunzip | sudo k3s ctr images import -'`
> (this one needs sudo; the registry loop above does not).

---

## 5 · Deploy the stack

```bash
kubectl apply -f deploy/k8s/            # RuntimeClass + namespace + ollama + tools + chat terminal
kubectl -n acoustic get pods -w         # wait for all Running
```

(If you were running Ollama on the host before, stop it so it's clearly the pod now:
`sudo systemctl disable --now ollama` — optional.)

---

## 6 · Pull the model into the Ollama pod

```bash
kubectl -n acoustic exec deploy/ollama -- ollama pull qwen2.5:3b
# verify GPU use while it runs a prompt:
kubectl -n acoustic exec deploy/ollama -- ollama run qwen2.5:3b "hello"
kubectl -n acoustic logs deploy/ollama | grep -i -E "cuda|gpu"   # should mention the GPU
```
(There's no `nvidia-smi` on Jetson; to watch load, run `sudo tegrastats` on the host during a prompt.)

---

## 7 · Chat with it

The front-end is a **web terminal** (ttyd) — no app, no login, no tool-wiring:

- **Open `http://<jetson-ip>:30088`** in any browser → the Acoustic Analyzer TUI. Ask
  *"what do you hear?"* → the agent calls `analyze_sound`, the DSP service answers, qwen explains
  it (synthetic 60 Hz hum while in demo mode).
- **Or from a shell:** `kubectl -n acoustic exec -it deploy/acoustic-chat -- python /app/shell/agent_shell.py`
- **Test the tool service directly** (NodePort `:30800`):
  ```bash
  curl -s http://<jetson-ip>:30800/healthz
  curl -s -X POST http://<jetson-ip>:30800/analyze -H 'Content-Type: application/json' -d '{"seconds":2}'
  ```

> **Edit the engine or the prompt:** the engine is `src/agent_shell.py` (ConfigMap `agent-shell`,
> shared by every vertical); the acoustic prompt is `deploy/agents/acoustic.prompt` (ConfigMap
> `acoustic-agent`). After a change, refresh the relevant ConfigMap and restart:
> ```bash
> kubectl -n acoustic create configmap agent-shell   --from-file=agent_shell.py=src/agent_shell.py       --dry-run=client -o yaml | kubectl apply -f -
> kubectl -n acoustic create configmap acoustic-agent --from-file=system.prompt=deploy/agents/acoustic.prompt --dry-run=client -o yaml | kubectl apply -f -
> kubectl -n acoustic rollout restart deploy/acoustic-chat
> ```

---

## 8 · When the mic is wired (later)

1. Plug in the USB audio adapter; confirm on the host: `arecord -l`.
2. In [`acoustic-tools.yaml`](../../deploy/k8s/acoustic-tools.yaml): uncomment the
   `securityContext.privileged`, the `/dev/snd` volume + mount, and set `ACOUSTIC_DEMO` to `"0"`.
3. `kubectl apply -f deploy/k8s/acoustic-tools.yaml`.

---

## 9 · Access from your Mac (and the rest of your LAN)

**Find the Jetson's IP:**
```bash
hostname -I | awk '{print $1}'        # e.g. 192.168.1.50
```

**The terminal UI — already exposed.** A NodePort binds on *all* node interfaces, so from your
Mac's browser just open `http://<jetson-ip>:30088`. Nothing else to do. (If the Jetson runs a
firewall: `sudo ufw allow 30088/tcp`.)

**Run `kubectl` from your Mac** (manage the cluster remotely):

1. On the **Jetson**, make the API cert valid for the LAN IP by adding a TLS SAN, then restart:
   ```bash
   curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--tls-san <jetson-ip>" sh -
   sudo systemctl restart k3s
   ```
   (k3s already adds the node's primary IP to the cert, so this is mainly needed if you'll reach
   it by a different name/IP — harmless to set regardless.)
2. On the **Mac**:
   ```bash
   brew install kubectl
   scp <user>@<jetson-ip>:/etc/rancher/k3s/k3s.yaml ~/.kube/jetson.yaml
   # point it at the Jetson instead of localhost (BSD sed on macOS):
   sed -i '' 's#https://127.0.0.1:6443#https://<jetson-ip>:6443#' ~/.kube/jetson.yaml
   export KUBECONFIG=~/.kube/jetson.yaml        # add to ~/.zshrc
   kubectl get nodes                            # should list the Jetson
   ```

**Give the Jetson a home DNS name (recommended).** A stable name beats chasing IPs and makes the
TLS cert clean. Pick one:

- **mDNS `.local` — zero setup:** Ubuntu runs avahi, so the Jetson answers to
  **`<its-hostname>.local`** on the LAN (macOS resolves `.local` natively). Name it `jetson` with
  `sudo hostnamectl set-hostname jetson` and it answers to `jetson.local`. Test: `ping jetson.local`
  from the Mac.
- **Router DHCP reservation + name (most robust):** reserve the Jetson's MAC to a fixed IP and name
  it in the router (e.g. `jetson`) so it resolves everywhere and survives reboots.
- **Pi-hole / dnsmasq:** add `address=/jetson.home/<jetson-ip>`.

Pin the IP (reservation or static) so the name keeps resolving. If you set `tls-san` at install
(§1), the cert already covers your name — done. **On an already-running cluster** that didn't,
add the names now (`config.yaml` is read at startup; the cert only regenerates after you remove
the dynamic cert):
```bash
# on the Jetson:
sudo tee /etc/rancher/k3s/config.yaml >/dev/null <<'EOF'
tls-san:
  - jetson.local
  - jetson
  - <jetson-ip>
EOF
# force the API serving cert to regenerate with the new names, then restart:
sudo rm -f /var/lib/rancher/k3s/server/tls/dynamic-cert.json
sudo systemctl restart k3s
# verify the names landed in the cert:
sudo openssl x509 -in /var/lib/rancher/k3s/server/tls/serving-kube-apiserver.crt \
  -noout -text | grep -A1 "Subject Alternative Name"
```
Now point the Mac's kubeconfig at the name and use the terminal by name:
```bash
sed -i '' 's#https://127.0.0.1:6443#https://jetson.local:6443#' ~/.kube/jetson.yaml
kubectl get nodes
# terminal UI:   http://jetson.local:30088
# tool service:  http://jetson.local:30800/healthz
```

**What's exposed on the LAN:** `acoustic-chat` (`:30088`, the terminal UI) and `acoustic-tools`
(`:30800`, the DSP API) are both NodePort — reachable from any device on your Wi-Fi. **Ollama
stays ClusterIP** (in-cluster only) — don't NodePort it; an open, unauthenticated LLM endpoint on
the LAN is best avoided.

> Note: the terminal UI has **no auth** — fine on a trusted home LAN. If you ever put this on an
> untrusted network, front it with something that authenticates, or keep it `exec`-only.

---

## kubectl cheat sheet

```bash
kubectl -n acoustic get pods,svc,pvc               # overview
kubectl -n acoustic logs deploy/ollama             # LLM logs (GPU/CUDA lines)
kubectl -n acoustic logs deploy/acoustic-tools     # tool service logs
kubectl -n acoustic rollout restart deploy/acoustic-tools   # after a rebuild+import
kubectl delete -f deploy/k8s/                       # tear down (PVCs persist unless deleted)
```
