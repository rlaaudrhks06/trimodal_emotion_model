"""HuggingFace 모델의 pytorch_model.bin을 safetensors로 변환해 로컬 폴더에 저장한다.

왜 필요한가: transformers 최신 버전은 CVE-2025-32434(torch.load 취약점) 때문에
torch 2.6 미만에서 `.bin` 로딩을 거부한다. 그런데 facebook/wav2vec2-* 계열은
safetensors를 제공하지 않아서, 서버(cu121에 맞춰둔 구버전 torch)에서 곧바로
`from_pretrained`를 못 쓴다.

torch를 올리는 건 잘 돌아가는 학습 환경 전체를 위험에 빠뜨리므로, 가중치만
안전한 형식으로 한 번 변환해 쓴다. 변환 자체는 transformers를 거치지 않고
torch.load(weights_only=True)로 읽으므로 차단에 걸리지 않는다.

실행 예:
    python scripts/convert_to_safetensors.py \\
        --repo facebook/wav2vec2-large-xlsr-53 \\
        --out models/wav2vec2-large-xlsr-53

이후 config의 audio.pretrained_model을 그 경로로 바꾸면 된다.
"""
import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

# from_pretrained가 가중치 외에 필요로 하는 파일들. 없는 것은 건너뛴다.
AUX_FILES = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bin-name", default="pytorch_model.bin")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[convert] {args.repo} 가중치 내려받는 중...")
    bin_path = hf_hub_download(args.repo, args.bin_name)

    # weights_only=True: 임의 코드 실행 위험이 있는 pickle 객체를 막고 텐서만 읽는다.
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    print(f"[convert] 텐서 {len(state)}개 로드")

    # safetensors는 저장소를 공유하는 텐서를 허용하지 않는다(wav2vec2는 weight_norm
    # 때문에 공유가 생길 수 있음). clone + contiguous로 각자 자기 메모리를 갖게 만든다.
    cleaned, shared = {}, 0
    seen = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        ptr = v.untyped_storage().data_ptr()
        if ptr in seen:
            shared += 1
        seen[ptr] = k
        cleaned[k] = v.detach().clone().contiguous()
    if shared:
        print(f"[convert] 저장소를 공유하던 텐서 {shared}개를 복사해 분리")

    save_file(cleaned, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"[convert] 저장: {out / 'model.safetensors'}")

    for name in AUX_FILES:
        try:
            p = hf_hub_download(args.repo, name)
            shutil.copy2(p, out / name)
            print(f"[convert]   + {name}")
        except Exception:
            pass  # 모델마다 없는 파일이 있는 게 정상

    print(f"\n[convert] 완료. config에서 아래 경로를 쓰면 된다:\n  {out}")


if __name__ == "__main__":
    main()
