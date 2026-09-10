#!/bin/bash
# 서브 세션용 worktree를 만든다 — **무거운 것들을 링크까지 걸어서.**
#
# [사건] 이 저장소는 .venv(1.8G)·models(1.2G)·checkpoints(4.8G)·data(27G)가 전부
#        gitignore다. `git worktree add`만 하면 새 트리엔 이것들이 **없고**,
#        테스트도 Stop 게이트도 돌지 않는다. 링크를 잊으면 "검증했다"가 거짓이 된다.
#
# 링크로 거는 이유: 복사하면 35GB가 늘고, worktree마다 캐시를 다시 만들게 된다.
#        읽기 전용으로만 쓰므로 공유가 안전하다.
#
# ⚠ checkpoints는 **쓰기도** 일어난다(학습 산출물). 서브 세션에서 학습을 돌리면
#   원본과 같은 곳에 쓴다. 그래서 **서브 세션에서는 학습을 돌리지 않는다**는 것이
#   규칙이다(CLAUDE.md 《서브 세션 절차》 ③).
#
# 사용:  bash .claude/hooks/worktree_new.sh <작업이름> [분기점]
# [폐기] 저장소가 데이터·가상환경을 git에 넣게 되면(그럴 일은 없다).

set -euo pipefail

NAME="${1:?사용: worktree_new.sh <작업이름> [분기점, 기본 main]}"
BASE="${2:-main}"

MAIN=$(git rev-parse --show-toplevel)
DEST="$(dirname "$MAIN")/$(basename "$MAIN")__$NAME"

if [ -e "$DEST" ]; then
  echo "이미 있다: $DEST" >&2
  exit 1
fi

git -C "$MAIN" worktree add -b "$NAME" "$DEST" "$BASE"

# gitignore라 따라오지 않는 무거운 것들을 링크한다.
LINKS=".venv models checkpoints data results/embeddings"
for d in $LINKS; do
  src="$MAIN/$d"
  [ -e "$src" ] || continue
  mkdir -p "$(dirname "$DEST/$d")"
  ln -s "$src" "$DEST/$d"
  printf "  링크 %-20s -> %s\n" "$d" "$src"
done

# ── 링크를 git이 무시하게 만든다. **이걸 빼면 public 저장소에 사고가 난다.**
#
# [사건] 첫 판이 이 단계를 빠뜨렸다. .gitignore의 패턴이 `checkpoints/`처럼 슬래시로
# 끝나는데, 슬래시는 **디렉터리만** 매칭한다. 심볼릭 링크는 git에게 디렉터리가 아니라
# 링크(mode 120000)라 패턴을 빠져나간다. 결과로 링크 5개가 미추적으로 떴고,
# `git add -A` 한 번이면 `/Users/<사용자>/...` 절대경로가 public 저장소에 커밋된다.
#
# .gitignore를 고치지 않는 이유: 그건 추적되는 규칙 파일이고 저장소 전체에 영향을 준다.
# 링크는 로컬 사정이므로 로컬 전용 파일에 적는 것이 맞다.
#
# worktree **전용** info/exclude는 안 먹는다 — git은 common 디렉터리 것을 읽는다.
# (이것도 시험해보고 알았다.)
COMMON=$(git -C "$MAIN" rev-parse --git-common-dir)
COMMON=$(cd "$(dirname "$COMMON")" && cd "$(basename "$COMMON")" && pwd)
mkdir -p "$COMMON/info"
if ! grep -qxF "results/embeddings" "$COMMON/info/exclude" 2>/dev/null; then
  {
    echo ""
    echo "# worktree_new.sh가 거는 심볼릭 링크. .gitignore의 `checkpoints/` 등은"
    echo "# 슬래시로 끝나 디렉터리만 매칭하므로 링크가 빠져나간다. 커밋되면"
    echo "# public 저장소에 로컬 절대경로가 남는다. 로컬 전용이라 여기에 둔다."
    for d in $LINKS; do echo "$d"; done
  } >> "$COMMON/info/exclude"
  echo "  exclude 등록  $COMMON/info/exclude"
fi

# 검증 — 조용히 실패하면 의미가 없다(규칙 4-1).
BAD=""
for d in $LINKS; do
  [ -e "$DEST/$d" ] || continue
  git -C "$DEST" check-ignore -q "$d" || BAD="$BAD $d"
done
if [ -n "$BAD" ]; then
  echo "  ⚠ 링크가 여전히 추적 대상이다:$BAD" >&2
  echo "    이대로 git add -A 하면 절대경로가 커밋된다. 커밋 전에 확인할 것." >&2
else
  echo "  검증  링크 전부 git 무시 확인"
fi

cat <<EOF

worktree 생성 완료
  경로     $DEST
  브랜치   $NAME  (분기점 $BASE)

여기서 하지 말 것
  · 학습·평가 실행 — checkpoints가 원본과 **같은 곳**을 가리킨다
  · 원본과 같은 파일 수정 — 그럴 거면 worktree를 쓸 이유가 없다

끝나면
  1) 테스트 통과 확인
  2) 커밋
  3) 머지는 사용자 승인 (가드가 ask 한다)
  4) git worktree remove $DEST
EOF
