"""Readable Colab entrypoint: check sources by default; --run opts into GPU execution.

The experiment modules are the exact expanded OneCell sources. Only transport
changes: visible source/ files replace the embedded compressed source capsule.
The original environment setup, experiment limits, and result ZIP recovery remain.
"""

import os, sys, json, time, threading, subprocess, zipfile, uuid, re, hashlib
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "launch-settings.json").read_text())
HF_SECRET_NAME = SETTINGS["HF_SECRET_NAME"]
EXPECTED_CAPS = SETTINGS["EXPECTED_CAPS"]
OUTPUT_LIMIT_BYTES = SETTINGS["OUTPUT_LIMIT_BYTES"]
ZIP_LIMIT_BYTES = SETTINGS["ZIP_LIMIT_BYTES"]
INITIAL_FREE_BYTES = SETTINGS["INITIAL_FREE_BYTES"]
EMBEDDED_SHA256 = json.loads((HERE / "source-manifest.json").read_text())


def read_checked_sources():
    source_root = HERE / "source"
    actual = {
        p.relative_to(source_root).as_posix()
        for p in source_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    }
    if actual != set(EMBEDDED_SHA256):
        raise ValueError("Source inventory mismatch")
    sources = {}
    for name, expected in EMBEDDED_SHA256.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Unsafe source path")
        path = source_root / p
        if path.is_symlink() or not path.resolve().is_relative_to(
            source_root.resolve()
        ):
            raise ValueError("Source symlink or escaped path")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Source digest mismatch: " + name)
        sources[name] = raw.decode("utf-8")
    return sources


def _loopkv_nb05_cell():
    import hashlib, shutil, signal, math

    run_id = (
        "loop-kv-nb05-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    work = Path("/content") / run_id
    work.mkdir(parents=True, exist_ok=False)
    root = work / "artifacts"
    root.mkdir()
    token = ""
    proc = None
    reader = None
    capture = work / "console.log"
    runtime_python = work / "runtime" / "bin" / "python"
    started = time.monotonic()
    outer_deadline = started + EXPECTED_CAPS["outer"]

    def clean(s):
        s = str(s)
        if isinstance(token, str) and token:
            s = s.replace(token, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", s)

    def save(name, obj):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
        os.replace(tmp, p)

    def load(name):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            return None

    def sources_plan_schedule(root):
        return (load("loop_kv_nb05_inputs.json") or {}).get("schedule", [])

    def stop():
        if proc is not None and proc.poll() is None:
            # This launcher owns the new process group, including any pip child.
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def evidence_files():
        return [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and not p.name.endswith(".tmp")
            and "__pycache__" not in p.parts
        ]

    try:
        # Read visible UTF-8 files; no encoded executable payload is used.
        sources = read_checked_sources()
        for name, body in sources.items():
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("Unsafe payload path")
            if hashlib.sha256(body.encode()).hexdigest() != EMBEDDED_SHA256[name]:
                raise RuntimeError("Source digest mismatch: " + name)
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        save("embedded_sha256.json", EMBEDDED_SHA256)
        save("nb05_capsule.json", EMBEDDED_SHA256)
        if shutil.disk_usage(work).free < INITIAL_FREE_BYTES:
            raise RuntimeError(
                "At least 30 GiB initial free disk is required for runtime, model weights, evidence and ZIP."
            )
        from google.colab import userdata

        token = userdata.get(HF_SECRET_NAME)
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("Enable Colab Secret HF_TOKEN for this notebook.")
        print(
            "NB05: Nanbeige actual-Q connection. Four new fixed texts; 540 model forwards including official gate, diagnostics, warmup, and 3 timing rounds. NSN3 full restoration versus GQA fusion. Evidence <=384 MiB; result ZIP <=256 MiB. No Drive or prior ZIP.",
            flush=True,
        )
        triton_cache = work / "triton_cache"
        triton_cache.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
        env.update(
            TRITON_CACHE_DIR=str(triton_cache),
            HF_TOKEN=token,
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DOWNLOAD_TIMEOUT="90",
            HF_HUB_ETAG_TIMEOUT="30",
            TOKENIZERS_PARALLELISM="false",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        env.update(LOOP_KV_NB05_PREPARE_STARTED=str(started), PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        save(
            "phase.json",
            dict(
                stage="PREPARE", deadline_monotonic=started + EXPECTED_CAPS["prepare"]
            ),
        )
        print(
            "Preparing a separate cu128 environment. Installation and the 7.77 GiB model download share a 30-minute preparation limit.",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, "-u", str(root / "loop_kv_nb05_runtime.py"), str(root)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        del env

        def drain():
            used = 0
            truncated = False
            with capture.open("wb") as f:
                for line in proc.stdout:
                    s = clean(line)
                    data = s.encode()
                    remaining = max(0, 1024 * 1024 - used)
                    if remaining:
                        kept = data[:remaining]
                        f.write(kept)
                        f.flush()
                        used += len(kept)
                        print(kept.decode(errors="replace"), end="", flush=True)
                    if len(data) > remaining and not truncated:
                        marker = b"\n[CONSOLE TRUNCATED AT 1 MiB; structured evidence retained]\n"
                        f.write(marker)
                        f.flush()
                        truncated = True
                        print(marker.decode(), flush=True)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        deadline = outer_deadline - EXPECTED_CAPS["zip"]
        while proc.poll() is None:
            time.sleep(1)
            phase = load("phase.json")
            active = load("active_case.json")
            expired = (
                active
                and active.get("status") == "STARTED"
                and time.monotonic() > active["deadline_monotonic"]
            )
            if (
                expired
                or (phase and time.monotonic() > phase["deadline_monotonic"])
                or time.monotonic() > deadline
                or sum(p.stat().st_size for p in evidence_files()) > OUTPUT_LIMIT_BYTES
            ):
                save(
                    "launcher_timeout.json",
                    dict(
                        reason="Case/phase/overall/evidence cap",
                        phase=phase,
                        active=active,
                    ),
                )
                stop()
                break
        reader.join(timeout=5)
        save(
            "launcher_status.json",
            dict(worker_exit_code=proc.returncode, run_id=run_id),
        )
    except BaseException as exc:
        stop()
        save(
            "launcher_error.json", dict(error_type=type(exc).__name__, error=clean(exc))
        )
        print(clean(exc), flush=True)
    finally:
        if reader is not None:
            reader.join(timeout=5)
        if capture.exists():
            shutil.copy2(capture, root / "console.log")
        interrupted = (root / "launcher_error.json").exists() or (
            root / "launcher_timeout.json"
        ).exists()
        verified = False
        terminal = load("run_status.json") or {}
        complete_claim = terminal.get("measurement_complete") is True
        if not complete_claim:
            save(
                "completion_check_skipped.json",
                dict(
                    reason="Worker did not claim complete measurements",
                    worker_status=terminal.get("status"),
                ),
            )
        if (
            complete_claim
            and not interrupted
            and proc is not None
            and proc.returncode == 0
        ):
            try:
                phase = load("phase.json") or {}
                left = (
                    min(
                        phase.get("deadline_monotonic", 0),
                        outer_deadline - EXPECTED_CAPS["zip"],
                    )
                    - time.monotonic()
                )
                if phase.get("stage") != "VERIFY" or left <= 0:
                    raise TimeoutError(
                        "Shared ten-minute verification window exhausted"
                    )
                check = subprocess.run(
                    [
                        str(runtime_python),
                        str(root / "verify_loop_kv_nb05.py"),
                        str(root),
                    ],
                    env={
                        **{
                            k: v
                            for k, v in os.environ.items()
                            if k
                            not in (
                                "PYTHONPATH",
                                "PYTHONHOME",
                                "HF_TOKEN",
                                "HUGGING_FACE_HUB_TOKEN",
                            )
                        },
                        "PYTHONNOUSERSITE": "1",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=left,
                )
                (root / "recovery_verification.log").write_text(
                    clean(check.stdout)[: 1024 * 1024]
                )
                verified = check.returncode == 0
            except BaseException as exc:
                save("verification_error.json", dict(error=clean(exc)))
        if not verified:
            for name in ["run_status.json", "summary.json"]:
                if (root / name).exists():
                    shutil.copy2(root / name, root / ("worker_" + name))
            progress = load("progress.json") or {}
            saved_steps = {
                p.parent.parent.name + "-" + p.name: len(list(p.glob("step-*.json")))
                for p in (root / "books").glob("b*/series/*")
                if p.is_dir()
            }
            recovery = dict(
                status="INCONCLUSIVE",
                experiment="LOOP-KV-NB05",
                measurement_complete=False,
                planned_forward_calls=540,
                attempted=progress.get("attempted", 0),
                finished=progress.get("finished", 0),
                saved_steps=saved_steps,
                missing_series=[
                    s["id"]
                    for s in sources_plan_schedule(root)
                    if saved_steps.get(s["id"], 0) < 9
                    or not (
                        root
                        / "books"
                        / ("b" + str(s["selected_book_rank"]))
                        / "series"
                        / s["id"].split("-", 1)[1]
                        / "complete.json"
                    ).is_file()
                ],
                active_case=load("active_case.json"),
                note="Complete result not independently verified. Original worker files and partial evidence retained.",
            )
            save("launcher_recovery.json", recovery)
            save("run_status.json", recovery)
            save("summary.json", recovery)

        def recovery_timeout(*_):
            raise TimeoutError("ZIP recovery time cap")

        previous_handler = signal.signal(signal.SIGALRM, recovery_timeout)
        signal.alarm(
            max(
                1,
                min(EXPECTED_CAPS["zip"], math.ceil(outer_deadline - time.monotonic())),
            )
        )
        try:
            hashes = {
                str(p.relative_to(root)): digest(p)
                for p in evidence_files()
                if p.name != "SHA256SUMS.json"
            }
            save("SHA256SUMS.json", hashes)
            evidence_size = sum(p.stat().st_size for p in evidence_files())
            if evidence_size > OUTPUT_LIMIT_BYTES:
                raise RuntimeError(
                    "Evidence exceeds 384 MiB; refusing to silently drop files or expand ZIP scope"
                )
            result = Path("/content") / (run_id + ".zip")
            temporary = result.with_suffix(".zip.tmp")
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3
            ) as z:
                for p in evidence_files():
                    z.write(p, str(p.relative_to(root)))
            if temporary.stat().st_size > ZIP_LIMIT_BYTES:
                raise RuntimeError("ZIP exceeds fixed 256 MiB cap")
            os.replace(temporary, result)
        except Exception as exc:
            print(
                "ZIP recovery failed:", clean(exc), "Files remain at", root, flush=True
            )
            return
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        print(
            "Closed NB05:",
            (load("summary.json") or {}).get("status", "INCONCLUSIVE"),
            flush=True,
        )
        print(
            "Result ZIP:",
            result,
            "(" + str(result.stat().st_size) + " bytes)",
            flush=True,
        )
        try:
            from google.colab import files

            files.download(str(result))
            print(
                "Download requested. The ZIP is also in the Colab Files panel.",
                flush=True,
            )
        except Exception as exc:
            print(
                "Download request failed:",
                clean(exc),
                "ZIP remains at",
                result,
                flush=True,
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        action="store_true",
        help="Run on a Colab GPU after reviewing source and limits",
    )
    args = ap.parse_args()
    sources = read_checked_sources()
    print(
        json.dumps(
            {
                "status": "SOURCES_VERIFIED",
                "files": len(sources),
                "model_execution": bool(args.run),
            }
        )
    )
    if args.run:
        if not Path("/content").is_dir():
            ap.error("This retained runtime targets Colab/Linux GPU, not a local Mac.")
        _loopkv_nb05_cell()
