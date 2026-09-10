#!/bin/bash
# Stop 훅 — 코드를 고쳐놓고 검증 없이 턴을 끝내지 못하게 한다.
#
# 근거: "'끝내기 전에 테스트를 돌려라'는 프롬프트에 적으면 **제안**이다.
#        Stop 훅이 멈추면 그건 **규칙**이다."  (커뮤니티 검증 패턴, 2026)
#
# [사건] 이 세션에서 "확인했습니다"를 여러 번 썼고 그중 하나는 확인 방법이 틀려
#        결론도 틀렸다(zsh 단어분리로 --help 6개가 '전부 실패'로 나옴).
#
# **한계 — 정직하게 적는다.** 이건 *완료* 게이트지 *정확성* 게이트가 아니다.
# "테스트를 돌렸는가"는 검사해도 "테스트가 옳은가"는 검사하지 못한다. 그 구멍은
# guard.py의 테스트 수정 ask 판정이 메운다.
#
# **worktree 주의**: 경로를 하드코딩하면 worktree에서 원본 트리를 검사하게 되어
# 엉뚱한 코드를 통과시킨다. 그래서 cwd에서 저장소 루트를 유도한다.
# [사건] 이 훅의 첫 판이 정확히 그 버그를 갖고 있었다.
#
# 비용: 코드가 깨끗하면 즉시 통과(수십 ms). 더러우면 융합 테스트 약 8초.
# [폐기] 스모크 테스트가 사라지거나 CI가 같은 역할을 하게 되면.

set -u

# 저장소 루트를 cwd에서 유도한다 — worktree에서도 자기 트리를 본다.
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$ROOT" || exit 0
[ -f "tests/test_forward_smoke.py" ] || exit 0     # 이 저장소가 아니면 관여 안 함

DIRTY=$(git status --porcelain -- src configs tests scripts 2>/dev/null | grep -v '^??' | head -20)
[ -z "$DIRTY" ] && exit 0

# .venv는 gitignore라 worktree엔 없다. 링크돼 있으면 쓰고, 없으면 안내만 하고 통과한다
# (검증을 못 하는 것과 검증이 실패한 것은 다르다 — 조용히 통과시키되 크게 알린다).
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  {
    echo "[Stop 게이트] ⚠ 검증을 **수행하지 못했다** — $ROOT/.venv 가 없다."
    echo "  worktree라면 .venv·models·checkpoints·data가 따라오지 않는다(전부 gitignore)."
    echo "  조치: bash .claude/hooks/worktree_new.sh <이름>  으로 만들면 링크까지 걸어준다."
    echo "  ※ 통과가 아니라 '검사 못 함'이다. 검증했다고 보고하지 말 것."
  } >&2
  exit 0
fi

OUT=$("$PY" - 2>&1 <<'PY'
import sys
sys.path.insert(0, '.')
try:
    from tests.test_forward_smoke import test_fusion_order, test_self_fusion_baseline
    test_fusion_order()
    test_self_fusion_baseline()
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)
PY
)

if [ $? -ne 0 ]; then
  {
    echo "[Stop 게이트] 코드를 고쳤는데 융합 스모크 테스트가 실패한다. 턴을 끝낼 수 없다."
    echo; echo "$OUT" | tail -20; echo
    echo "고친 파일:"; echo "$DIRTY" | sed 's/^/  /'
    echo
    echo "이 테스트들은 '플래그가 아무 일도 안 하는데 통과하는 것'을 잡으려고 만든 것이다."
    echo "통과시키려고 테스트를 무르게 하지 말 것."
  } >&2
  exit 2
fi

echo "[Stop 게이트] 융합 스모크 통과($ROOT). 커밋 전에 전체 스모크도 돌릴 것:" >&2
echo "  .venv/bin/python tests/test_forward_smoke.py" >&2
exit 0
