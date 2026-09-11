#!/bin/bash
# 작업 폴더 **전체**의 파일 명부를 찍고, 이전 명부가 있으면 비교한다.
#
# 쓰는 자리: checkout·merge·rebase **전에** 한 번, **후에** 한 번.
#
#   scripts/tree_census.sh before     # 명부를 저장
#   git merge ...
#   scripts/tree_census.sh after      # 저장된 명부와 비교, 사라진 파일을 보여준다
#
# [사건] "docs/ 22개"만 세고 머지했다. 그 전엔 "디스크에 있다"를 머지 **전**에만
#        확인하고 머지 후에 다시 보지 않아 문서 14개가 사라진 걸 늦게 알았다.
#        git은 추적하던 파일을 제거하는 커밋으로 옮기면 작업트리에서도 지우므로,
#        지켜야 할 것이 docs/ 하나라고 가정하면 다른 곳에서 같은 일이 난다.
#        그래서 특정 폴더가 아니라 **전체**를 센다.
#
# 거대 디렉터리(데이터·가상환경·가중치)는 개수만 센다 — 216만 개 경로를 나열하면
# 몇 분이 걸리고 diff도 못 읽는다. 이 디렉터리들은 .gitignore라 git이 건드릴 수
# 없지만, 그래도 개수는 확인한다(전제가 틀렸을 때 잡히도록).

set -u
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "git 저장소 아님"; exit 1; }
cd "$ROOT" || exit 1
MODE="${1:-before}"
DIR="${TMPDIR:-/tmp}/tree_census_$(basename "$ROOT")"
mkdir -p "$DIR"

# 개수만 셀 디렉터리(거대). 나머지는 경로를 전부 나열한다.
COUNT_ONLY='data .venv models checkpoints checkpoints_periodic checkpoints_periodic_si_w2v archived_runs results/embeddings'

census() {
  local out="$1"
  {
    echo "## 거대 디렉터리 (개수만)"
    for d in $COUNT_ONLY; do
      [ -d "$d" ] && printf "%-32s %s\n" "$d" "$(find "$d" -type f 2>/dev/null | wc -l | tr -d ' ')"
    done
    echo
    echo "## 나머지 전체 경로"
    prune_args=()
    for d in $COUNT_ONLY; do prune_args+=(-path "./$d" -o); done
    find . \( "${prune_args[@]}" -path ./.git \) -prune -o -type f -print 2>/dev/null | LC_ALL=C sort
  } > "$out"
}

case "$MODE" in
  before)
    census "$DIR/before.txt"
    echo "[census] 명부 저장: 나열 $(grep -c '^\./' "$DIR/before.txt")개 + 거대 $(sed -n '/^## 거대/,/^$/p' "$DIR/before.txt" | grep -cE ' [0-9]+$')곳 -> $DIR/before.txt"
    ;;
  after)
    [ -f "$DIR/before.txt" ] || { echo "[census] before 명부가 없다 — 먼저 'before'로 찍을 것"; exit 1; }
    census "$DIR/after.txt"
    gone=$(comm -23 <(grep '^\./' "$DIR/before.txt") <(grep '^\./' "$DIR/after.txt"))
    new=$(comm -13 <(grep '^\./' "$DIR/before.txt") <(grep '^\./' "$DIR/after.txt"))
    big=$(diff <(sed -n '/^## 거대/,/^$/p' "$DIR/before.txt") <(sed -n '/^## 거대/,/^$/p' "$DIR/after.txt") || true)
    echo "[census] 전 $(grep -c '^\./' "$DIR/before.txt")개 -> 후 $(grep -c '^\./' "$DIR/after.txt")개"
    if [ -n "$gone" ]; then
      echo "[census] ⚠ 사라진 파일 $(echo "$gone" | wc -l | tr -d ' ')개:"; echo "$gone" | sed 's/^/    /'
    else
      echo "[census] ✅ 사라진 파일 없음"
    fi
    [ -n "$new" ] && echo "[census] 새 파일 $(echo "$new" | wc -l | tr -d ' ')개 (머지로 들어온 것이면 정상)"
    if [ -n "$big" ]; then
      echo "[census] ⚠ 거대 디렉터리 개수 변동:"; echo "$big" | sed 's/^/    /'
    else
      echo "[census] ✅ 거대 디렉터리 개수 그대로"
    fi
    [ -z "$gone" ] && [ -z "$big" ]
    ;;
  *) echo "사용법: $0 before|after"; exit 2 ;;
esac
