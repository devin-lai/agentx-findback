"""Sign and verify the FindBack Agent Skills with OpenSSF model signing (OMS).

Every skill directory -- the portable `skills/<name>` directories and each internal
`src/agentx/skills/<name>` -- receives a detached `skill.oms.sig` covering every file in that
directory, the layout used by the NVIDIA Verified Skills catalog. Signing uses an ECDSA P-256
private key kept under the git-ignored `.private/` directory; verification needs only the
published public key, `skills/findback-skills.pub`.

`model-signing` is deliberately not a project dependency. Run this script with an interpreter
that has it, for example a throwaway virtual environment:

    python3 -m venv /tmp/sigvenv && /tmp/sigvenv/bin/pip install "model-signing==1.1.1"
    /tmp/sigvenv/bin/python scripts/ops/sign_skills.py verify

Commands:
    keygen   create the key pair once; refuses to overwrite an existing key
    sign     write skill.oms.sig into every skill directory, then verify it
    verify   verify every signature strictly; exit 1 on any failure

A signature proves that a directory is byte-for-byte what was signed. It does not prove that a
skill is safe or correct; read its skill card and evaluation record for that.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIGNATURE = "skill.oms.sig"
DEFAULT_KEY = ROOT / ".private" / "signing" / "findback-skills-signing.key"
DEFAULT_PUBLIC_KEY = ROOT / "skills" / "findback-skills.pub"
SKILL_ROOTS = ("skills", "src/agentx/skills")
PASSWORD_ENV = "FINDBACK_SIGNING_KEY_PASSWORD"
# Machine-local files that a clean checkout will not contain. Signing one would make every
# verification after `git clone` fail, so their presence is an error, not something to sign.
STRAY_NAMES = {".DS_Store", "Thumbs.db", "__pycache__"}
STRAY_SUFFIXES = (".pyc", ".pyo", ".swp", ".bak", "~")


class SigningError(RuntimeError):
    pass


def skill_dirs(root: Path = ROOT) -> list[Path]:
    """Every directory directly under a skill root that holds a SKILL.md."""
    found = []
    for name in SKILL_ROOTS:
        base = root / name
        if base.is_dir():
            found.extend(
                sorted(p for p in base.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
            )
    return found


def stray_files(directory: Path) -> list[str]:
    """Files that must not be signed because a clean checkout would not have them."""
    problems = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_symlink():
            problems.append(f"{relative} (symlink)")
        elif any(part in STRAY_NAMES for part in relative.parts) or path.name.endswith(
            STRAY_SUFFIXES
        ):
            problems.append(str(relative))
    return problems


def git_ignored(paths: list[Path], root: Path = ROOT) -> list[str]:
    """Paths that Git would ignore, so a clone would not contain them. Empty without Git."""
    if not paths:
        return []
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(str(p) for p in paths),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def fingerprint(public_key_path: Path) -> str:
    """SHA-256 of the PEM public key: the key hint model-signing stores in each bundle."""
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_public_key(public_key_path.read_bytes())
    pem = key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return hashlib.sha256(pem).hexdigest()


def signed_files(signature_path: Path) -> dict[str, str]:
    """The file digests a signature covers, read from its in-toto statement."""
    import base64

    bundle = json.loads(signature_path.read_text())
    statement = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    return {r["name"]: r["digest"] for r in statement["predicate"]["resources"]}


def _model_signing():
    try:
        import model_signing
    except ImportError:
        raise SigningError(
            "model-signing is not installed for this interpreter. Use a separate environment: "
            'python3 -m venv /tmp/sigvenv && /tmp/sigvenv/bin/pip install "model-signing==1.1.1"'
            " && /tmp/sigvenv/bin/python scripts/ops/sign_skills.py verify"
        ) from None
    return model_signing


def _hashing_config(model_signing):
    # The signature file itself is never part of what it signs; Git metadata is ignored as in
    # the model_signing CLI and the NVIDIA catalog signatures.
    return model_signing.hashing.Config().set_ignored_paths(
        paths=[SIGNATURE], ignore_git_paths=True
    )


def verify_directory(directory: Path, public_key: Path) -> tuple[bool, str]:
    """Strictly verify one skill directory: unsigned additions fail as well as changes."""
    model_signing = _model_signing()
    directory = directory.resolve()  # an installed skill is often a symlink to its source
    signature = directory / SIGNATURE
    if not signature.is_file():
        return False, f"missing {SIGNATURE}"
    try:
        model_signing.verifying.Config().use_elliptic_key_verifier(
            public_key=public_key
        ).set_hashing_config(_hashing_config(model_signing)).verify(directory, signature)
    except Exception as exc:  # model_signing raises ValueError and crypto errors alike
        return False, str(exc) or type(exc).__name__
    return True, f"{len(signed_files(signature))} files"


def keygen(args) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key_path, public_path = args.key.resolve(), args.public_key.resolve()
    for path in (key_path, public_path):
        if path.exists():
            raise SigningError(
                f"{path} already exists. Rotating the key is a deliberate act: move the old "
                "key pair away yourself, then re-sign every skill."
            )
    if key_path.is_relative_to(ROOT) and not git_ignored([key_path.relative_to(ROOT)]):
        raise SigningError(f"Refusing to write a private key where Git would track it: {key_path}")
    private = ec.generate_private_key(ec.SECP256R1())
    password = os.environ.get(PASSWORD_ENV)
    encryption = (
        serialization.BestAvailableEncryption(password.encode())
        if password
        else serialization.NoEncryption()
    )
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(
            private.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption
            )
        )
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    print(
        json.dumps(
            {
                "private_key": str(key_path),
                "private_key_encrypted": bool(password),
                "public_key": str(public_path),
                "public_key_sha256": fingerprint(public_path),
            },
            indent=2,
        )
    )
    return 0


def sign(args) -> int:
    model_signing = _model_signing()
    key_path = args.key.resolve()
    if not key_path.is_file():
        raise SigningError(f"No private key at {key_path}. Run `keygen` once first.")
    directories = [d.resolve() for d in args.skill] or skill_dirs()
    problems = {}
    for directory in directories:
        found = stray_files(directory)
        if directory.is_relative_to(ROOT):
            files = [p.relative_to(ROOT) for p in directory.rglob("*") if p.is_file()]
            found += [f"{p} (ignored by Git)" for p in git_ignored(files)]
        if found:
            problems[str(directory)] = found
    if problems:
        raise SigningError(
            "Remove machine-local files before signing; a clean checkout would not have them: "
            + json.dumps(problems)
        )
    signer = model_signing.signing.Config().use_elliptic_key_signer(
        private_key=key_path, password=os.environ.get(PASSWORD_ENV)
    )
    failed = 0
    for directory in directories:
        signer.set_hashing_config(_hashing_config(model_signing)).sign(
            directory, directory / SIGNATURE
        )
        ok, detail = verify_directory(directory, args.public_key.resolve())
        failed += not ok
        print(f"{'SIGNED' if ok else 'FAIL'}  {_display(directory)}  ({detail})")
    return 1 if failed else 0


def verify(args) -> int:
    public_key = args.public_key.resolve()
    if not public_key.is_file():
        raise SigningError(f"No public key at {public_key}.")
    directories = list(args.skill) or skill_dirs()
    if not directories:
        raise SigningError("No skill directories found.")
    results = []
    for directory in directories:
        ok, detail = verify_directory(directory, public_key)
        results.append({"skill": _display(directory), "verified": ok, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {_display(directory)}  ({detail})")
    failures = sum(not r["verified"] for r in results)
    print(
        f"{len(results) - failures}/{len(results)} signatures verified with public key "
        f"sha256:{fingerprint(public_key)}"
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 1 if failures else 0


def _display(directory: Path) -> str:
    resolved = directory.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("keygen", keygen), ("sign", sign), ("verify", verify)):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
        if name != "verify":
            command.add_argument("--key", type=Path, default=DEFAULT_KEY)
        if name != "keygen":
            command.add_argument(
                "--skill",
                type=Path,
                action="append",
                default=[],
                help="Skill directory to process (repeatable). Default: every skill in the repo.",
            )
        if name == "verify":
            command.add_argument("--json", type=Path, help="Also write the results to this file.")
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except SigningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
