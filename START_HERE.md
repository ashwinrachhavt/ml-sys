# 🚀 START HERE - Quick Setup Guide

## You're seeing errors? Here's the fix!

You have a **global virtual environment** active that conflicts with this project. Follow these 3 steps:

### Step 1: Deactivate Your Virtual Environment

```bash
deactivate
```

### Step 2: Check System Status

```bash
./check_system.sh
```

This will tell you exactly what's wrong and how to fix it.

### Step 3: Run End-to-End Tests

```bash
./test_e2e.sh
```

This will verify everything works correctly.

## ✅ Success Looks Like This

When everything is working, you'll see:

```
✨ All critical tests passed! System is working correctly. ✨

You can now:
  • Start the API: uv run uvicorn mlsys.inference.service:app --reload
  • Run full stack: docker-compose up -d
  • View API docs: http://localhost:8000/docs
```

## 📚 What to Read Next

| Document | What It Does | When to Use It |
|----------|--------------|----------------|
| **HOW_TO_TEST.md** | Complete testing guide | When you want to verify everything works |
| **QUICKSTART.md** | 5-minute getting started | Quick overview of commands |
| **README.md** | Full documentation | Complete reference |
| **SETUP.md** | Detailed setup steps | Step-by-step installation |
| **TROUBLESHOOTING.md** | Fix common issues | When something breaks |
| **MLOPS_ARCHITECTURE.md** | System design | Understand the architecture |

## 🛠️ Helpful Scripts

| Script | What It Does |
|--------|--------------|
| `./check_system.sh` | Check what's working |
| `./fix_env.sh` | Fix environment issues |
| `./test_e2e.sh` | Run all tests |
| `./setup.sh` | Complete setup |

## 🎯 Common Commands

Once your environment is fixed:

```bash
# Run tests
uv run pytest

# Start API
uv run uvicorn mlsys.inference.service:app --reload

# Train model
uv run python scripts/train.py --config config/base_config.yaml

# Full stack (API + MLflow + Prometheus + Grafana)
docker-compose up -d
```

## ❓ Still Having Issues?

1. Make sure you ran `deactivate`
2. Run `./check_system.sh` to diagnose
3. Run `./fix_env.sh` to fix environment
4. Read `TROUBLESHOOTING.md` for specific errors
5. Check that uv is installed: `uv --version`

## 🎉 What You Get

This is a **production-grade MLOps system** with:

- ✅ XGBoost model with SMOTE for class imbalance
- ✅ FastAPI inference service with Prometheus metrics
- ✅ Automated CI/CD with GitHub Actions
- ✅ Scheduled weekly model retraining
- ✅ MLflow experiment tracking
- ✅ Docker containerization
- ✅ Full observability stack (Prometheus + Grafana)
- ✅ Comprehensive testing suite
- ✅ Pre-commit hooks for code quality

---

**TL;DR:** Run `deactivate` then `./test_e2e.sh` to verify everything works!
