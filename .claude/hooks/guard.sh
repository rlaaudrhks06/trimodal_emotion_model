#!/bin/bash
# guard.py 실행 래퍼 — **가드가 못 돌면 통과가 아니라 차단한다(fail-closed).**
#
# [사건] guard.py를 `python3 guard.py`로 등록했더니 훅이 셸과 다른 PATH로 실행돼
#        /usr/bin/python3(3.9.6)를 잡았고, PEP 604(`str | None`)에서 TypeError로 죽었다.
#        종료 코드가 1이라 **차단(2)이 아니었고, 모든 도구 호출이 그냥 통과했다.**
#        가드가 등록돼 있는데 아무것도 막지 않는 상태였다 — `git branch -D`도,
#        규칙 파일 수정도 전부 통과했다. 호출 로그를 붙여 "한 번도 불리지 않았다"를
#        확인하고서야 원인을 찾았다.
#
#        이것이 CLAUDE.md에 적어둔 "가드가 조용히 통과하는 것이 가장 나쁜 실패"의
#        실제 사례이고, 하필 가드 자신에게서 났다.
#
# 그래서 두 겹으로 막는다.
#   ① 인터프리터를 직접 고른다 — PATH에 의존하지 않는다
#      (guard.py는 `from __future__ import annotations`로 3.9에서도 돈다)
#   ② 그래도 실행이 실패하면 **deny를 내보낸다.** 안전 장치가 고장났는데 조용히
#      열어두느니 막는다.
#
# 복구: 가드가 오작동해 작업이 막히면 ~/.claude/settings.json 의 PreToolUse 항목을
#       지운다. 막히는 것이 조용히 통과하는 것보다 낫다.
# [폐기] guard.py가 사라지면.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="$HERE/guard.py"
ERR="${TMPDIR:-/tmp}/guard_err.txt"

deny() {
  # 메시지를 JSON 문자열에 넣기 전에 깨뜨릴 문자를 없앤다.
  # [사건] 파이썬 SyntaxError 메시지에 `"`가 들어 있어 JSON이 중간에 끊겼다.
  #        exit 2가 이겨서 차단 자체는 유효했지만 **왜 막혔는지가 화면에 안 떴다.**
  #        가드가 고장난 상황에서 사유를 못 보면 복구가 늦어진다.
  msg=$(printf '%s' "$1" | tr -d '"\\' | tr '\n\r\t' '   ')
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"[가드 고장] %s  가드를 실행하지 못해 안전을 보장할 수 없다. 조용히 통과시키지 않는다. 복구: ~/.claude/settings.json 의 PreToolUse 항목을 확인할 것."}}\n' "$msg"
  exit 2
}

[ -f "$GUARD" ] || deny "guard.py를 찾을 수 없다: $GUARD"

PY=""
for c in /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3 2>/dev/null)" /usr/bin/python3; do
  if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; break; fi
done
[ -n "$PY" ] || deny "실행 가능한 python3이 없다"

IN=$(cat)
OUT=$(printf '%s' "$IN" | "$PY" "$GUARD" 2>"$ERR")
RC=$?

# guard.py는 통과=0, 차단=2 만 낸다(ask도 0 + JSON). 그 밖의 종료 코드는 **죽은 것**이다.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 2 ]; then
  deny "guard.py가 종료 코드 $RC 로 죽었다: $(head -c 200 "$ERR" 2>/dev/null | tr '\n' ' ')"
fi

[ -n "$OUT" ] && printf '%s\n' "$OUT"
exit "$RC"
