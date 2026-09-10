#!/usr/bin/env python3
"""PreToolUse 가드 — 글로 적은 규칙은 안 지켜지므로 기계가 막는다.

근거(Claude Code 공식 문서):
  · CLAUDE.md는 "컨텍스트로 취급하지, 강제되는 설정으로 취급하지 않는다.
    무엇을 결정하든 상관없이 동작을 막으려면 PreToolUse 훅을 써라."
  · `--dangerously-skip-permissions`는 **권한 프롬프트만** 건너뛴다. 훅은 못 건너뛴다.
    그래서 훅이 마지막 방어선이다.
  · 출력 규약: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
    "permissionDecision": "deny"|"ask"|"allow", "permissionDecisionReason": "..."}}
    종료 코드 2도 차단하지만, JSON을 쓰면 사유가 그대로 전달되고 "ask"를 쓸 수 있다.

판정 순서 — 위에서부터, 처음 걸리는 것이 이긴다.

  파일 수정                                       Bash 명령
  ─────────────────────────────────────           ────────────────────────────────
  1. 개인키·비밀       deny                        A. 개인키 노출        deny
  2. 규칙 파일         deny (승인 토큰 있으면 통과)   B. 되돌릴 수 없는 git deny
  3. 테스트 약화       ask  (사람이 판단)            C. push / merge       ask
  4. 문서              allow                        D. 파이프가 exit 삼킴  deny
  5. main에서 코드     deny

**가드 자체가 고장나면 통과시킨다**(exit 0). 가드 버그로 작업이 멈추면 안 된다.
단, 비밀·파괴 판정은 예외 없이 문자열 검사만 하므로 고장날 여지가 거의 없다.

[폐기] 각 항목의 [폐기] 주석 참고. 프로젝트가 끝나면 전체가 죽는다.
"""
# **파이썬 3.9 호환 필수.** 훅은 셸과 다른 PATH로 실행돼 /usr/bin/python3(3.9)를
# 잡는다. PEP 604(`str | None`)는 3.10+에서만 런타임 동작하므로 이 줄이 없으면
# import 시점에 TypeError로 죽고, **exit 2가 아니라 1이라 도구가 통과한다.**
# [사건] 실제로 그렇게 등록됐고 가드가 아무것도 막지 않았다. 로그를 붙여서야 알았다.
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# 경로를 하드코딩하지 않는다 — 스크립트 위치에서 유도한다.
# [사건] 첫 판이 /Users/<사용자>/... 를 박아둬서 public 저장소에 올릴 수 없었고,
#        저장소가 옮겨지면 조용히 무력화될 구조였다.
HOOK_DIR = Path(__file__).resolve().parent          # <repo>/.claude/hooks
REPO = HOOK_DIR.parents[1]                          # <repo>
ROOT = REPO.parent                                  # 작업 폴더(키가 있는 곳)

# ── 규칙을 이루는 파일. 고치는 것 = 규칙을 바꾸는 것이므로 사용자 승인이 필요하다.
RULE_FILES = {REPO / "CLAUDE.md", ROOT / "CLAUDE.md", REPO / ".gitignore"}
RULE_DIRS = {REPO / ".claude" / "hooks", REPO / ".claude" / "rules",
             ROOT / ".claude" / "hooks", ROOT / ".claude" / "rules"}
APPROVAL = REPO / ".claude" / ".rule-change-approved"      # 1회용 토큰

# ── 문서는 언제든 고칠 수 있어야 한다(사용자 결정 2026-09-10).
#    확장자가 아니라 경로로 판정한다 — docs/ 아래의 pdf·png도 문서다.
DOC_DIRS = ("/docs/",)
DOC_SUFFIXES = {".md"}

# ── 테스트: 통과시키려고 테스트를 약화시키면 검증 계층이 통째로 무너진다.
TEST_MARKERS = ("/tests/", "test_", "_test.")

# ── 비밀: 이 저장소는 public이고 서버 개인키가 작업 폴더에 있다.
SECRET_PAT = re.compile(r"\.pem\b|\.key\b|id_rsa|id_ed25519|BEGIN (OPENSSH|RSA|EC) PRIVATE")

# ── 되돌릴 수 없는 git. 이 프로젝트는 v11 체크포인트가 유일본이고 30GB가 로컬에만 있다.
DESTRUCTIVE_GIT = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(?:"
    r"reset\s+--hard|clean\s+-[a-z]*f|checkout\s+--?\s*\.|restore\s+\.|"
    r"push\s+.*--force(?!-with-lease)|filter-repo|filter-branch|"
    r"branch\s+-D|update-ref\s+-d|reflog\s+expire"
    r")"
)

# ── 검증을 건너뛰는 우회 수단. `-n`은 오탐이 많아 git commit 문맥에서만 잡는다.
BYPASS = re.compile(
    r"--no-verify\b|--no-gpg-sign\b|"
    r"\bgit\s+(?:-C\s+\S+\s+)?commit\b[^|;&]*\s-\w*n\b|"
    r"\bHUSKY=0\b|\bSKIP=\S|--dangerously-skip-permissions\b"
)

# 파이프로 넘길 때 실패가 의미 있는 명령. ls·cat·echo까지 막으면 너무 시끄러워서
# 규칙 자체가 무시당한다.
FAILURE_MATTERS = ("python", "pytest", "git ", "pip ", "make ", "npm ", "node ", "ssh ")


def decide(kind: str, reason: str) -> None:
    """kind: deny | ask. 결정과 사유를 JSON으로 내보내고 끝낸다."""
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": kind,
        "permissionDecisionReason": reason,
    }}, ensure_ascii=False))
    # JSON만으로도 차단되지만, 스키마가 바뀌어도 막히도록 deny는 exit 2를 겸한다.
    # (가드가 조용히 통과하는 것이 이 프로젝트에서 가장 나쁜 실패다 — 규칙 4-1)
    sys.exit(2 if kind == "deny" else 0)


def is_rule_file(p: Path) -> bool:
    """절대경로로만 보면 **worktree의 규칙 파일을 놓친다.**

    [사건] 이 함수의 첫 판이 원본 트리 경로만 알고 있어서, worktree에서 CLAUDE.md를
    고치는 것이 무방비였다. 서브 세션 절차를 도입하면서 발견했다.
    그래서 이름·구조로도 판정한다 — 어느 트리에 있든 규칙은 규칙이다.
    """
    if p in RULE_FILES or any(d in p.parents for d in RULE_DIRS):
        return True
    parts = p.parts
    if ".claude" in parts:
        i = parts.index(".claude")
        if len(parts) > i + 1 and parts[i + 1] in ("hooks", "rules"):
            return True
    return p.name in ("CLAUDE.md", ".gitignore")


def is_doc(p: Path) -> bool:
    s = str(p)
    return any(d in s for d in DOC_DIRS) or p.suffix.lower() in DOC_SUFFIXES


def is_test(p: Path) -> bool:
    s = str(p)
    return any(m in s for m in TEST_MARKERS)


def branch_of(p: Path) -> str | None:
    d = p if p.is_dir() else p.parent
    try:
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def check_edit(ti: dict) -> None:
    raw = ti.get("file_path") or ti.get("notebook_path")
    if not raw:
        return
    p = Path(raw)
    p = p.resolve() if p.is_absolute() else p

    # 1. 개인키를 파일로 쓰려는 시도
    #    [사건] 서버 개인키가 작업 폴더에 있고 저장소는 public이다.
    #    [폐기] 서버 회수(2026-11-05) 이후.
    if SECRET_PAT.search(str(p)):
        decide("deny",
               f"개인키로 보이는 파일을 쓰려 한다: {p.name}\n"
               f"키는 사람이 직접 배치한다. 저장소는 public이고, 한 번 들어가면 파일을 "
               f"지워도 키를 재발급해야 한다.")

    # 2. 규칙 파일 — 승인 토큰이 있으면 1회만 통과시키고 토큰을 태운다.
    if is_rule_file(p):
        if APPROVAL.exists():
            try:
                APPROVAL.unlink()
            except Exception:
                pass
            return
        decide("deny",
               f"규칙 파일을 고치려 한다: {p.name}\n"
               f"규칙이 작업을 방해하면 고치는 게 맞지만, 고칠지는 사용자가 정한다.\n"
               f"절차: ① 무엇이 막혔고 왜 규칙이 틀렸는지 1~2줄 보고\n"
               f"      ② 사용자가 실행:  touch {APPROVAL}\n"
               f"      ③ 다시 시도 (토큰 1회용)   ④ [사건]·[폐기] 갱신   ⑤ 결과 보고")

    # 3. 테스트 수정 — 막지 않고 **묻는다**. 정당한 수정도 많다.
    #    [사건] 이 저장소의 스모크 테스트는 "플래그가 무동작인 채 통과하는 것"을 잡으려고
    #    만들었다. 통과시키려고 테스트를 무르게 하면 검증 계층이 통째로 무너진다.
    #    [폐기] 없음.
    if is_test(p):
        decide("ask",
               f"테스트를 고치려 한다: {p.name}\n"
               f"기능을 검증하려는 수정인가, 통과시키려는 완화인가?\n"
               f"이 저장소의 스모크 테스트는 '플래그가 아무 일도 안 하는데 통과하는 것'을 "
               f"잡으려고 만든 것이라, 무르게 하면 검증이 통째로 무의미해진다.")

    # 4. 문서 — 통과
    if is_doc(p):
        return

    # 5. main에서 코드 수정
    #    [사건] 이 프로젝트의 실패는 커밋 시점에 정상으로 보였다.
    #    [폐기] 없음.
    if branch_of(p) in ("main", "master"):
        decide("deny",
               f"main에서 코드를 고치려 한다: {p.name}\n"
               f"작업은 브랜치를 따고 시작한다. main은 검증된 상태로 두고 언제든 되돌아갈 "
               f"수 있어야 한다.\n"
               f"조치: git checkout -b <작업이름> main\n"
               f"예외: 문서(docs/ 아래 전부, 모든 *.md)는 main에서 바로 고쳐도 된다.")


def check_bash(ti: dict) -> None:
    cmd = (ti.get("command") or "").strip()
    if not cmd:
        return

    # 0. **가드 자체를 우회하려는 시도**
    #    [사건] anthropics/claude-code #40117 — Claude Code가 CLAUDE.md의 명시적 금지에도
    #    불구하고 `--no-verify`·stash·quiet 플래그로 pre-commit 훅을 **6번 연속** 우회했다.
    #    프롬프트로 적은 금지는 우회당한다. 우회 자체를 기계가 막아야 한다.
    #    [폐기] 없음.
    if BYPASS.search(cmd):
        decide("deny",
               "가드·검증을 우회하는 플래그가 들어 있다.\n"
               "--no-verify · git commit -n · HUSKY=0 · SKIP= 등은 검증을 건너뛴다.\n"
               "실제로 이 우회가 문서화된 사고가 있다(claude-code #40117: 명시적 금지에도 "
               "6번 연속 우회).\n"
               "검증이 방해가 되면 우회하지 말고, 무엇이 왜 막혔는지 보고한다 — "
               "규칙이 틀렸으면 규칙 변경 절차를 탄다.")

    # A. 개인키를 읽거나 옮기거나 출력하려는 명령
    if SECRET_PAT.search(cmd) and not re.search(r"\bchmod\b|\bls\b|--help", cmd):
        decide("deny",
               "명령에 개인키 경로/내용이 들어 있다.\n"
               "키는 읽거나 복사하거나 출력하지 않는다. 저장소는 public이고, 접속에는 "
               "경로만 넘기면 된다(`ssh -i <키>`는 chmod·ls와 함께일 때만 허용).")

    # B. 되돌릴 수 없는 git
    #    [사건] v11 체크포인트가 유일본이고 30GB가 로컬에만 있다. 미커밋 작업이 날아가면
    #    복구 수단이 없다. 이 저장소는 예전에 filter-repo로 히스토리를 다시 쓴 적이 있다.
    #    [폐기] 없음.
    if DESTRUCTIVE_GIT.search(cmd):
        decide("deny",
               "되돌릴 수 없는 git 명령이다.\n"
               "reset --hard · clean -f · checkout . · push --force · filter-repo 등은 "
               "미커밋 작업과 히스토리를 지운다.\n"
               "조치: 무엇을 왜 지우려는지 보고하고, 지울 것과 지킬 것의 관계를 먼저 적는다. "
               "되돌릴 방법이 있으면 그쪽을 쓴다(stash · branch · --force-with-lease).")

    # C. push / merge — 하드 차단이 아니라 **사용자에게 묻는다**
    #    [사건] 머지·공개 시점은 사람이 정한다는 합의(2026-09-10). push는 되돌릴 수 없고
    #    저장소는 public이다.
    #    [폐기] 저장소가 private이 되고 사용자가 위임하면.
    m = re.search(r"\bgit\s+(?:-C\s+\S+\s+)?(push|merge)\b", cmd)
    if m and "--dry-run" not in cmd:
        decide("ask",
               f"git {m.group(1)} — 시점은 사용자가 정한다.\n"
               f"push는 되돌릴 수 없고(이 저장소는 public), merge는 main의 '검증된 상태'를 "
               f"바꾼다. 무엇을 왜 하려는지 확인하고 승인한다.")

    # D. 종료 코드를 삼키는 파이프 ( a || b 의 ||는 파이프가 아니다 )
    #    [사건] ImportError로 죽은 실행이 `| tail` 때문에 exit 0으로 보고됐다.
    #    [폐기] 없음.
    if "pipefail" not in cmd and "|" in cmd.replace("||", ""):
        if any(k in cmd.split("|")[0] for k in FAILURE_MATTERS):
            decide("deny",
                   "파이프가 앞 명령의 종료 코드를 삼킨다.\n"
                   "조치: 명령 앞에 `set -o pipefail; ` 를 붙인다.\n"
                   "그리고 종료 코드만으로 성공을 판정하지 않는다 — 출력을 직접 읽는다.")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input") or {}
    try:
        if tool in ("Edit", "Write", "NotebookEdit"):
            check_edit(ti)
        elif tool == "Bash":
            check_bash(ti)
    except SystemExit:
        raise
    except Exception:
        return 0        # 가드 버그로 작업을 막지 않는다
    return 0


if __name__ == "__main__":
    sys.exit(main())
