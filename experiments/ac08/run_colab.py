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
EXPECTED_MAIN_TEMPLATE = SETTINGS["EXPECTED_MAIN_TEMPLATE"]
EXPECTED_BASE_SCHEDULE = SETTINGS["EXPECTED_BASE_SCHEDULE"]
EXPECTED_CAPS = SETTINGS["EXPECTED_CAPS"]
OUTPUT_LIMIT_BYTES = SETTINGS["OUTPUT_LIMIT_BYTES"]
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


def _loopkv_ac08_cell():
    import hashlib, shutil

    run_id = (
        "loop-kv-ac08-"
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
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        os.replace(tmp, p)

    def load(name):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            return None

    def stop():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    try:
        # Read visible UTF-8 files; no encoded executable payload is used.
        sources = read_checked_sources()
        for name, body in sources.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("Unsafe payload path")
            if hashlib.sha256(body.encode()).hexdigest() != EMBEDDED_SHA256[name]:
                raise RuntimeError("Source digest mismatch: " + name)
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        save("embedded_sha256.json", EMBEDDED_SHA256)
        save("ac08_capsule.json", EMBEDDED_SHA256)
        save("planned_schedule_template.json", EXPECTED_MAIN_TEMPLATE)
        if shutil.disk_usage(work).free < 2 * OUTPUT_LIMIT_BYTES:
            raise RuntimeError(
                "At least 12 GiB free disk is required for evidence plus ZIP, in addition to model storage."
            )
        from google.colab import userdata

        token = userdata.get(HF_SECRET_NAME)
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("Enable Colab Secret HF_TOKEN for this notebook.")
        print(
            "AC08: native / packed full-V restoration / packed KV-fused model comparison. 5 route gates + 6 full warmups; 4 new books x 3 arms x 2 repeats x 255 predictions; continuation ends at 1023, with 3 additional compression boundaries. HF_TOKEN, no Drive or prior ZIP.",
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
        env.update(LOOP_KV_AC08_PREPARE_STARTED=str(started), PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        save(
            "phase.json",
            dict(
                stage="PREPARE", deadline_monotonic=started + EXPECTED_CAPS["prepare"]
            ),
        )
        print(
            "Preparing a separate cu128 environment; package download is included in the 30-minute preparation limit.",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, "-u", str(root / "loop_kv_ac08_runtime.py"), str(root)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        del env

        def drain():
            with capture.open("w") as f:
                for line in proc.stdout:
                    s = clean(line)
                    f.write(s)
                    f.flush()
                    print(s, end="", flush=True)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        deadline = min(
            outer_deadline - EXPECTED_CAPS["independent_verify"] - EXPECTED_CAPS["zip"],
            time.monotonic()
            + sum(
                EXPECTED_CAPS[k]
                for k in ("prepare", "gate", "warmup", "main", "worker_verify")
            ),
        )
        while proc.poll() is None:
            time.sleep(1)
            phase = load("phase.json")
            active = load("active_case.json")
            expired = (
                active
                and active.get("status") == "STARTED"
                and time.monotonic() > active["deadline_monotonic"] + 15
            )
            if (
                expired
                or (phase and time.monotonic() > phase["deadline_monotonic"] + 15)
                or time.monotonic() > deadline
                or sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                > OUTPUT_LIMIT_BYTES
            ):
                save(
                    "launcher_timeout.json",
                    dict(reason="Case/phase/overall cap", phase=phase, active=active),
                )
                stop()
                break
        reader.join(timeout=10)
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
            reader.join(timeout=10)
        if capture.exists():
            shutil.copy2(capture, root / "console.log")
        interrupted = (root / "launcher_error.json").exists() or (
            root / "launcher_timeout.json"
        ).exists()
        verified = False
        terminal = load("run_status.json") or {}
        complete_claim = terminal.get("status") in ("COMPLETE_COMPARISON",)
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
                check = subprocess.run(
                    [
                        str(runtime_python),
                        str(root / "verify_loop_kv_ac08.py"),
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
                    timeout=max(
                        1,
                        min(
                            EXPECTED_CAPS["independent_verify"],
                            outer_deadline - time.monotonic() - EXPECTED_CAPS["zip"],
                        ),
                    ),
                )
                (root / "recovery_verification.log").write_text(clean(check.stdout))
                verified = check.returncode == 0
            except BaseException as exc:
                save("verification_error.json", dict(error=clean(exc)))
        if not verified:
            for name in ["run_status.json", "summary.json"]:
                if (root / name).exists():
                    shutil.copy2(root / name, root / ("worker_" + name))
            observed = []
            malformed = 0
            for p in sorted((root / "main").glob("*/result.json")):
                try:
                    observed.append(json.loads(p.read_text())["id"])
                except (OSError, ValueError, KeyError, TypeError):
                    malformed += 1
            main_schedule = []
            selection_error = None
            try:
                lock = load("selection_lock.json")
                if lock is not None:
                    if lock.get("status") != "FROZEN_BEFORE_MODEL_LOAD":
                        raise ValueError("Selection lock not frozen")
                    if set(lock.get("files", {})) != {
                        "materials.json",
                        "schedule.json",
                        "selection.jsonl",
                    }:
                        raise ValueError("Incomplete selection lock")
                    for name, digest in lock["files"].items():
                        if (
                            hashlib.sha256((root / name).read_bytes()).hexdigest()
                            != digest
                        ):
                            raise ValueError("Changed frozen input")
                    books = lock["selected_ids"]
                    allowed = {
                        c["book_id"]: c["candidate_index"]
                        for c in load("loop_kv_ac08_candidates.json")["candidates"]
                    }
                    if (
                        len(books) != len(set(books))
                        or len(books) != 4
                        or not set(books) <= set(allowed)
                        or books != sorted(books, key=allowed.__getitem__)
                    ):
                        raise ValueError("Invalid selected books")
                    for template in EXPECTED_MAIN_TEMPLATE:
                        sp = {
                            k: v
                            for k, v in template.items()
                            if k not in ("id_template", "selected_book_rank")
                        }
                        sp["book_id"] = books[template["selected_book_rank"]]
                        sp["id"] = template["id_template"].format(book_id=sp["book_id"])
                        main_schedule.append(sp)
                    if load("schedule.json") != main_schedule:
                        raise ValueError(
                            "Resolved schedule differs from fixed template"
                        )
            except BaseException as exc:
                main_schedule = []
                selection_error = clean(exc)
            all_schedule = EXPECTED_BASE_SCHEDULE + main_schedule
            completed_all = []
            for sp in all_schedule:
                rr = load(sp["phase"] + "/" + sp["id"] + "/result.json")
                if (
                    rr
                    and rr.get("id") == sp["id"]
                    and rr.get("status") == "COMPLETE_FINITE"
                ):
                    completed_all.append(sp["id"])
            recovery = dict(
                completed_all_ids=completed_all,
                missing_all_ids=[
                    sp["id"] for sp in all_schedule if sp["id"] not in completed_all
                ],
                status="INCONCLUSIVE",
                experiment="LOOP-KV-AC08",
                planned=24,
                recorded_ids=observed,
                missing_ids=[x["id"] for x in main_schedule if x["id"] not in observed],
                unassigned_main_slots=24 if not main_schedule else 0,
                selection_error=selection_error,
                malformed_lines=malformed,
                primary_verdicts={
                    key: "NOT_EVALUABLE" for key in ("H1", "H2", "H3", "H4")
                },
                active_case=load("active_case.json"),
                legacy_full_p0_pass=False,
                note="Complete result not verified. Original worker files and partial measurements retained; unresolved book slots are counted without inventing book IDs.",
            )
            save("launcher_recovery.json", recovery)
            save("run_status.json", recovery)
            save("summary.json", recovery)
        import signal

        def recovery_timeout(*_):
            raise TimeoutError(
                "ZIP recovery capped at 300 seconds / remaining outer budget"
            )

        previous_handler = signal.signal(signal.SIGALRM, recovery_timeout)
        signal.alarm(
            max(1, min(EXPECTED_CAPS["zip"], int(outer_deadline - time.monotonic())))
        )
        try:
            hashes = {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*"))
                if p.is_file()
                and p.name != "SHA256SUMS.json"
                and not p.name.endswith(".tmp")
                and "__pycache__" not in p.parts
            }
            save("SHA256SUMS.json", hashes)
            result = Path("/content") / (run_id + ".zip")
            temporary = result.with_suffix(".zip.tmp")
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3
            ) as z:
                for p in sorted(root.rglob("*")):
                    if (
                        p.is_file()
                        and not p.name.endswith(".tmp")
                        and "__pycache__" not in p.parts
                    ):
                        z.write(p, str(p.relative_to(root)))
            os.replace(temporary, result)
        except Exception as exc:
            print(
                "ZIP recovery failed:", clean(exc), "Files remain at", root, flush=True
            )
            return
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        print("Result ZIP:", result, flush=True)
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
        _loopkv_ac08_cell()
