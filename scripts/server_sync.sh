#!/bin/bash
# 서버 작업 트리를 origin의 <브랜치>로 맞춘다. 실행 전에 이걸 돌리고, 마지막 줄의 해시를 본다.
#
#   scripts/server_sync.sh main            # origin/main으로
#   scripts/server_sync.sh audio-probe     # 다른 브랜치로
#
# [사건] 같은 충돌이 사흘에 세 번 났다. 서버에서 평가가 결과 JSON·CSV를 만들고, 그걸 Mac으로
#        회수해 커밋·push하면, 서버에는 **같은 내용의 미추적 파일**이 남는다. 그 상태에서
#        merge/checkout을 하면 git이 "덮어쓰겠다"며 거부한다. 매번 손으로 md5를 비교하고
#        지웠다 — 세 번째부터는 스크립트다.
#
# 미추적 결과 파일이 origin의 것과 **바이트 단위로 같을 때만** 지운다. 다르면 멈춘다 —
# 회수 안 된 결과를 지우면 평가를 다시 돌려야 한다.
set -u
BR="${1:?브랜치 이름}"
cd "$(git rev-parse --show-toplevel)" || exit 1
git fetch -q origin "$BR" || { echo "❌ fetch 실패"; exit 1; }
diff_files=0
for f in $(git status --porcelain results/ configs/ scripts/ src/ tests/ 2>/dev/null | grep '^??' | awk '{print $2}'); do
  if git cat-file -e "origin/$BR:$f" 2>/dev/null; then
    if [ "$(md5sum "$f" | cut -c1-32)" = "$(git show "origin/$BR:$f" | md5sum | cut -c1-32)" ]; then
      rm -f "$f"
    else
      echo "⚠ 미추적 $f 가 origin/$BR 의 것과 다르다 — 회수 안 된 결과? 중단"; diff_files=1
    fi
  fi
done
[ "$diff_files" = 0 ] || exit 1
git checkout -q "$BR" 2>/dev/null || git checkout -q -b "$BR" "origin/$BR"
git merge -q --ff-only "origin/$BR" || { echo "❌ ff 머지 실패 — 서버에 로컬 커밋이 있나?"; git status --short | head; exit 1; }
echo "서버 @ $(git rev-parse --short HEAD) ($BR) · origin과 $([ "$(git rev-parse HEAD)" = "$(git rev-parse origin/$BR)" ] && echo 일치 || echo 불일치)"
