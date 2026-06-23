---
name: rocm-driver-recovery
description: Runbook for recovering a shared MI300X (gfx942) box after the ROCm + amdgpu driver stack was uninstalled — diagnose hardware-vs-software, re-add AMD apt repos, reinstall amdgpu-dkms + ROCm userspace, load and verify with rocm-smi.
source: Written from the 2026-06-23 incident on banff-ccs-aus-p19-05 (8x MI300X), where a CI harness ran its ROCm teardown phase and never reinstalled. Steps performed and verified live during recovery.
---


Recovering a shared MI300X box after the ROCm + `amdgpu` stack was uninstalled
out from under it. Written after the 2026-06-23 incident on
`banff-ccs-aus-p19-05`, where a CI harness ran its teardown phase
(`modprobe -r amdgpu`, removed apt repos, purged `amdgpu-dkms` + ROCm
userspace) and never ran the reinstall phase, leaving the GPUs unusable.

Host facts for this incident: Ubuntu 22.04 (jammy), kernel
`5.15.0-174-generic`, 8x Instinct MI300X (PCI device id `0x74a1`),
prior stack `rocm-core 7.2.4` + AMD-internal driver `amdgpu-dkms 6.19.10`.

---

## 1. Diagnose: is it hardware or software?

The cards being gone from `rocm-smi` does **not** mean a hardware failure.
Check, in order:

```bash
# GPU tools present / driver loaded?
rocm-smi                      # "command not found" => userspace purged
ls -la /dev/kfd               # missing => kfd driver not loaded
lsmod | grep -iE 'amdgpu|amdkfd'   # empty => module not loaded

# Is the hardware still on the bus? (this is the key check)
lspci | grep -iE 'Processing accelerators|Instinct'
#   8x "Aqua Vanjaram [Instinct MI300X]" => hardware is FINE, software problem
```

If `lspci` shows the cards, it is a software/driver problem — proceed.

### Confirm it was a clean unload, not a crash

```bash
sudo dmesg -T | grep -iE 'amdgpu|kfd|ras|ecc|xgmi|reset|panic' | tail -60
```

A clean unload looks like orderly `finishing device` / `ttm finalized` lines
for every PCI device, with **no** RAS/ECC/XGMI fault, GPU reset, or panic.
That means a human/script ran `modprobe -r amdgpu`, not a hardware fault.

### Find who did it (shared box)

```bash
# Audit the sudo log for the teardown commands
sudo journalctl _COMM=sudo | grep -iE 'modprobe -r amdgpu|autoremove|dpkg --purge|amdgpu-dkms' | tail -40

# All sudo activity for a suspected user, with dates
sudo journalctl _COMM=sudo | grep -i '<user>' | grep 'COMMAND=' | tail -40
```

Look for the signature: `modprobe -r amdgpu` → `rm .../rocm*.list` →
`apt-get autoremove amdgpu-dkms` → `dpkg --purge ... rocblas hipblaslt ...`.
On a shared machine, **contact that user before reinstalling** — their CI
may own the teardown and resume on its own. Only reinstall once you've
decided not to wait.

---

## 2. Decide scope

A bare `modprobe amdgpu` is **not** enough if the stack was purged:

- The in-tree Ubuntu `amdgpu.ko` (`5.15.0-174-generic`) does **not** support
  MI300X. Verify: `modinfo amdgpu | grep -i 74a1` => no match. Loading it
  gives you `/dev/kfd` but **zero GPU nodes** (only the CPU node in KFD
  topology), no `renderD*` nodes, and `dmesg` stops at "Add CPU node".
- The MI300X-capable driver came from `amdgpu-dkms`; the math/runtime libs
  (`rocblas`, `hipblaslt`, `rocminfo`, `rocm-smi`, `hsa-rocr`, `comgr`, ...)
  came from the ROCm repo. Both were removed, so both must be reinstalled.

So the real fix = re-add AMD apt repos + reinstall driver + userspace.

---

## 3. Pre-flight checks

```bash
# Repos still configured? (the teardown deletes these)
ls /etc/apt/sources.list.d/
grep -riE 'rocm|amdgpu' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null
apt-cache policy amdgpu-dkms        # Candidate: (none) => no repo, must re-add

# DKMS build prerequisites
which dkms
ls -d /usr/src/linux-headers-$(uname -r)    # headers must exist to build

# Network reachable?
curl -sS -o /dev/null -w "%{http_code}\n" https://repo.radeon.com/amdgpu/

# OS identity (codename + arch) — needed for repo lines
. /etc/os-release && echo "$VERSION_CODENAME $VERSION_ID"   # jammy 22.04
dpkg --print-architecture                                  # amd64
```

### Pick matching versions

The previous userspace was `rocm-core 7.2.4`, so target ROCm **7.2.4**.
Confirm the repo paths exist before adding them:

```bash
# userspace lives under rocm/apt/<ver>
curl -sS -o /dev/null -w "%{http_code}\n" https://repo.radeon.com/rocm/apt/7.2.4/dists/jammy/Release   # expect 200

# The standalone amdgpu/<ver> repo only carries ROCm-tied driver builds up to
# ~7.0.3 publicly. For 7.1+/7.2 the driver pairs come from amdgpu/latest.
# Check the dkms+firmware version pair there:
curl -sS https://repo.radeon.com/amdgpu/latest/ubuntu/dists/jammy/main/binary-amd64/Packages \
  | grep -A2 -E '^Package: amdgpu-dkms(-firmware)?$' | grep -E 'Package|Version'
```

Note: the prior driver `6.19.10` (build suffix `2343748`) was an AMD
**internal** build and is not on the public repo. The public `amdgpu/latest`
driver (e.g. `6.16.13` + firmware `30.30.4`) is compatible with ROCm 7.2.4
userspace and supports MI300X. If you specifically need the internal build,
you must use AMD's internal repo instead.

---

## 4. Add repos + GPG key

```bash
sudo mkdir -p --mode=0755 /etc/apt/keyrings
curl -sS https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Driver repo (latest amdgpu, jammy)
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/latest/ubuntu jammy main' \
  | sudo tee /etc/apt/sources.list.d/amdgpu.list

# Userspace repo (ROCm 7.2.4, jammy)
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 jammy main' \
  | sudo tee /etc/apt/sources.list.d/rocm.list

# Pin so ROCm packages win over distro ones
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  | sudo tee /etc/apt/preferences.d/rocm-pin-600

sudo apt-get update
```

---

## 5. Install driver + userspace

A `dpkg --purge` teardown usually leaves **broken dependency state**
(dangling `hip-dev`, `hip-runtime-amd`, etc.). Clear it first, then install:

```bash
sudo apt-get --fix-broken install -y      # resolves leftover purge state
sudo apt-get install -y amdgpu-dkms       # builds the DKMS kernel module (slow)
sudo apt-get install -y rocm-smi-lib rocminfo   # minimal userspace to verify
# (or the full set you need: rocm-libs hipblaslt rocblas miopen-hip rccl ...)
```

`amdgpu-dkms` triggers a DKMS build for the running kernel and regenerates
the initramfs — it can take a few minutes. Success ends with the per-module
`Installing to /lib/modules/<kver>/updates/dkms/` lines and `depmod`.

---

## 6. Load the driver + verify

```bash
sudo modprobe amdgpu
sleep 8

ls -la /dev/kfd                                   # device node back
ls /dev/dri/ | grep -c renderD                    # 8 GPUs => >=8 renderD* nodes
ls /sys/class/kfd/kfd/topology/nodes/             # node 0 = CPU, 1..8 = GPUs

# dmesg should show per-GPU init, not just the CPU node
sudo dmesg | grep -iE 'amdgpu' | grep -iE 'Initialized amdgpu|VRAM|ring' | tail

# Authoritative check: rocm-smi talks to all cards via the driver
/opt/rocm-7.2.4/bin/rocm-smi
```

A healthy `rocm-smi` lists all 8 devices (`0x74a1`) with sane temps/power and
full power caps. That is proof the cards are up.

### Gotchas observed

- **PATH:** ROCm tools live in `/opt/rocm-7.2.4/bin`, which is not on `PATH`
  by default — use full paths or add it. `rocminfo` may report 0 GPUs if run
  with a stale `LD_LIBRARY_PATH`; trust `rocm-smi` via the driver instead.
- **Shared box / CI:** if a CI runner service exists
  (e.g. `actions.runner.*`), it may tear the stack down again or install its
  own pinned version. Tell the owner you reinstalled.

---

## 7. (If aborting instead) Restore the clean state

If you decide to wait for the owner rather than reinstall, undo any partial
work to leave the box exactly as found:

```bash
sudo modprobe -r amdgpu                                   # unload any stub module
sudo dpkg --remove --force-remove-reinstreq amdgpu-dkms   # if half-unpacked
```

---

## Appendix: incident root-cause signature (2026-06-23)

From `journalctl _COMM=sudo`, the teardown sequence that caused the outage:

```
05:54:05  modprobe -r amdgpu
05:54:15  rm -rf /etc/apt/sources.list.d/{rocm-build,amdgpu-build}.list
05:54:18  apt-get -y autoremove amdgpu-dkms
05:54:52  apt-get -y autoremove rocm-opencl ; rm /etc/rocm_metadata.conf
05:54:59  apt-get -y autoremove rocm-libs miopen-hip rocm-bandwidth-test rccl
05:55:14  dpkg --purge --force-all rocblas hipblaslt hsa-rocr-dev rocminfo comgr rocm-device-libs ...
```

No reinstall phase followed — that is what left the GPUs down.
