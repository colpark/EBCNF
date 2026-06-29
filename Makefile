.PHONY: ffinr test1 train-smoke train clean
PY ?= python3

# GPU selection: `make train DEVICE=cuda:1` (or export NF_DEVICE=cuda:1 / CUDA_VISIBLE_DEVICES=1)
DEVICE ?=
DEVFLAG := $(if $(strip $(DEVICE)),--device $(DEVICE),)

# Diagnosis: a vanilla Fourier-feature INR on the raw ±1 event field (shows why it's hard).
ffinr:
	$(PY) -m ebcnf.event_inr

# Test 1: EvINR-style event->intensity reconstruction (SIREN + derivative supervision).
test1:
	$(PY) -m ebcnf.evinr

# Amortized LAINR-EBC: tiny end-to-end sanity (CPU, seconds).
train-smoke:
	$(PY) -m ebcnf.train_amortized --smoke $(DEVFLAG)

# Amortized LAINR-EBC: reasonable simulated dataset (auto-detects GPU).
train:
	$(PY) -m ebcnf.train_amortized --config configs/ebc_lainr.yaml $(DEVFLAG)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
