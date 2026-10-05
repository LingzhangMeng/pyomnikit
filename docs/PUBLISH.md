# Publishing `pyomnikit`

The repo is already a git repository with a first commit. Below are the exact steps to put it on
**GitHub** and (optionally) **PyPI**. Replace `LingzhangMeng` with your GitHub username.

> This machine has **no `gh` (GitHub CLI)** and **no `twine`/`build`** installed. Per the project
> rule, install any missing tool **yourself** (commands below use the Aliyun mirror for speed in CN).

---

## 1. GitHub — create the repo and push

**a) Create the empty repo** at <https://github.com/new> → name **`pyomnikit`** (already confirmed
free on GitHub: 0 existing repos), *Public*, **do not** initialise with README/.gitignore (the
local repo already has them).

**b) Push** (HTTPS; use a Personal Access Token as the password, or SSH):

```bash
cd /media/meng/Bioinformatics/Lupus/pyomnikit
git branch -M main
git remote add origin https://github.com/LingzhangMeng/pyomnikit.git
git push -u origin main
```

If push is slow from China, either use SSH (`git@github.com:LingzhangMeng/pyomnikit.git`, needs an
SSH key on GitHub) or a proxy:
```bash
git remote set-url origin https://ghproxy.net/https://github.com/LingzhangMeng/pyomnikit.git
```

**c) Nice-to-haves on the repo page:** add the description *"Depth-robust program scoring,
cross-species conservation, spatial-niche and cross-omics integration for kidney-disease
transcriptomics"*, topics `bioinformatics single-cell multi-omics spatial-transcriptomics
lupus-nephritis`, and confirm the CI badge (`.github/workflows/ci.yml`) goes green.

---

## 2. PyPI — publish the package

The distribution name **`pyomnikit` is free**; the import stays `omnikit`.

```bash
# install the build tools (manual, once)
/home/meng/envs/sidish_env/bin/python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ build twine

# build sdist + wheel
cd /media/meng/Bioinformatics/Lupus/pyomnikit
/home/meng/envs/sidish_env/bin/python -m build

# upload (needs a PyPI API token: https://pypi.org/manage/account/token/)
/home/meng/envs/sidish_env/bin/python -m twine upload dist/*
#   username: __token__   password: pypi-<your-token>

# test the release
python -m pip install --upgrade pyomnikit
```

**TestPyPI first (recommended):**
```bash
/home/meng/envs/sidish_env/bin/python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ pyomnikit
```

---

## 3. Versioning / release

- Bump `version` in `pyproject.toml` (and, for a matching GitHub release, tag it):
  ```bash
  git tag v0.1.0 && git push origin v0.1.0
  ```
- **Zenodo DOI** (optional, for a citable release): connect the GitHub repo to Zenodo once; each
  tagged release then gets a DOI — put it in `README.md` and `CITATION.cff`.

---

## 4. What is already done locally

- Repo assembled at `pyomnikit/` (`src/omnikit/`, `tests/smoke.py`, `examples/`, `docs/MATH.md`,
  `README.md`, `LICENSE` (MIT), `.gitignore`, `CITATION.cff`, `.github/workflows/ci.yml`).
- Wheel built and installed in a clean venv; `tests/smoke.py` passes from the **installed artifact**.
- git repo initialised with a first commit.
