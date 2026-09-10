#!/bin/bash
# SessionStart 훅 — 세션 시작 때마다 "지금 어디에 있는지"를 주입한다.
# stdout이 그대로 컨텍스트로 들어간다.
#
# [사건] 서버 회수(11/5 17:00)가 이번 기간 최대 위험인데 매 세션 날짜를 다시
#        계산하고 있었다. 브랜치 상태도 매번 git status로 확인했다.
#
# **네트워크 호출 금지.** SSH로 서버 상태를 확인하려다 세션 시작이 네트워크에
# 묶이면 안 된다. 로컬에서 즉시 계산되는 것만 넣는다.
#
# [폐기] 2026-11-05 회수 이후 D-day 줄은 죽는다. 그때 이 훅을 지우거나 고친다.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 스크립트 위치에서 유도
DEADLINE="2026-11-05 17:00"

# D-day — macOS date와 GNU date 양쪽에서 동작하게
if secs=$(date -j -f "%Y-%m-%d %H:%M" "$DEADLINE" +%s 2>/dev/null) \
   || secs=$(date -d "$DEADLINE" +%s 2>/dev/null); then
  left=$(( (secs - $(date +%s)) / 86400 ))
  echo "[일정] GPU 서버 자동 회수까지 D-${left} (${DEADLINE}, 사전 공지 없음·복구 불가)"
  echo "       결선이 11/3~11/5라 회수가 결선 마지막 날이다. 백업은 결선 출발 전에 끝낸다."
fi

if [ -d "$REPO/.git" ]; then
  b=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)
  n=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  ahead=$(git -C "$REPO" rev-list --count main.."$b" 2>/dev/null || echo 0)
  echo "[코드] 브랜치 ${b} (main 대비 +${ahead}커밋) · 미커밋 ${n}건"
  [ "$b" = "main" ] && echo "       ⚠ main이다. 코드를 고치려면 먼저 브랜치를 판다(가드가 막는다)."
fi

echo "[규칙] 상세는 ${REPO}/CLAUDE.md — 검증·비교·데이터 규칙이 거기 있다."
