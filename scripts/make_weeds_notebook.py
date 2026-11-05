"""Generate notebooks/colab_weeds_pilot.ipynb with the src/weeds modules embedded, so that the notebook runs on Colab
without pushing code to GitHub. Rerun after any change in src/weeds:  python scripts/make_weeds_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['src/__init__.py'] + [f'src/weeds/{m}.py' for m in ('__init__', 'data', 'model', 'train', 'decision', 'analyze')]
ZIP_URL = 'https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip'
ZIP_BYTES = 336989804  # size served by GFZ on 2026-10-29

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n').splitlines(keepends=True)})


files = {m: (ROOT / m).read_text(encoding='utf-8') for m in MODULES}

md('''
# Thí nghiệm thử: WeedsGalore, phân vùng cỏ dại và quyết định phun thuốc

Mục đích: kiểm tra đề tài có khả thi không trước khi đầu tư tiếp. Thí nghiệm gồm ba phần:
1. Huấn luyện DeepLabv3+ dùng ảnh 5 dải phổ, bài toán 3 lớp (nền, ngô, cỏ), theo thiết lập của Celikkan và cộng sự (WACV 2025).
2. Đánh giá trên tập test và so với kết quả trong bài báo (mIoU 82,90%).
3. Tầng quyết định: chia mỗi ô ảnh thành các ô lưới, quyết định **phun / không phun / chưa quyết** bằng conformal. Phương án chính hiệu chỉnh chéo 2 phần theo vị trí trên tập test (4 vị trí hiệu chỉnh cho 4 vị trí còn lại và ngược lại), vì tập val đã dùng để chọn mô hình; hiệu chỉnh trên val là phương án phụ.

**Lần chạy 2:** batch 8 như bài báo (lần 1 dùng batch 2, là mặc định trong code của tác giả), mọi thiết lập khác giữ nguyên lần 1. Kết quả lưu vào thư mục mới `run_dlv3p_msi_seed0_b8`, không ghi đè lần 1 (`run_dlv3p_msi_seed0`). Dữ liệu gọn của lần 1 trên Drive được dùng lại, nên mục 2 sẽ tự bỏ qua.

**Tiêu chí đạt/không đạt** (đề xuất, sửa được trong `src/weeds/analyze.py`): mIoU test ≥ 77,90; lượng thuốc giảm ≥ 30% so với phun toàn ruộng; tỉ lệ ô "chưa quyết" ≤ 30%. Các tiêu chí này tính ở thiết lập chính: ô lưới 0,5 m, ngưỡng cỏ 1% (ngưỡng minh họa), α = β = 0,1, gộp 3 đợt bay trước khi phun thuốc.

**Cách chạy nhanh nhất:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Ở mục 1, Colab hỏi quyền truy cập Google Drive: bấm đồng ý. Phần tải và chuẩn bị dữ liệu (mục 2) tốn thêm khoảng 5–10 phút GPU. Nếu muốn tiết kiệm hạn mức GPU, chạy mục 1–2 trên runtime CPU trước, rồi đổi sang T4 và Run all; mục 2 sẽ tự bỏ qua vì dữ liệu đã có trên Drive.

- Nếu một bước bị lỗi, "Run all" sẽ dừng ở bước đó. Hãy gửi thông báo lỗi cho Claude.
- Huấn luyện tự dừng an toàn trước khoảng 3,5 giờ và lưu checkpoint lên Drive. Nếu chưa xong, ở phiên sau chạy lại mục 1, 3 rồi 5 để chạy tiếp.
- Giữ tab Colab mở trong lúc chạy, thỉnh thoảng xem qua, vì Colab có thể ngắt kết nối khi trình duyệt không hoạt động lâu.

Kết quả nằm trong `MyDrive/De_Tai_Nong_San/weeds_pilot/run_dlv3p_msi_seed0_b8/`. Mục cuối tạo file `ket_qua_weeds_pilot_b8.zip` để tải về.
''')

md('## 1. Kết nối Drive và ghi mã nguồn (chạy ở **mọi** phiên, cả CPU lẫn GPU)')
code(f'''
import os, sys, json
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/De_Tai_Nong_San/weeds_pilot')
DATA_DRIVE = BASE / 'data'
BATCH = 8  # run 2: the paper's batch size; run 1 used 2 (authors' code default) and lives in run_dlv3p_msi_seed0
RUN = BASE / f'run_dlv3p_msi_seed0_b{{BATCH}}'  # a new folder, so run 1's train_done.json cannot skip training
RESULTS = Path(f'results/weeds_pilot_b{{BATCH}}')  # folder inside the downloaded zip; run 1 is in results/weeds_pilot
for p in (DATA_DRIVE, RUN):
    p.mkdir(parents=True, exist_ok=True)

WORK = Path('/content/weeds_pilot')
FILES = {json.dumps(files, ensure_ascii=False)}
for rel, text in FILES.items():
    path = WORK / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print('Đã ghi', len(FILES), 'file mã nguồn vào', WORK)
print('Batch', BATCH, '| thư mục kết quả:', RUN)

import subprocess
def run(*args):
    """Run a module of the pilot, stream its output, and stop 'Run all' if it fails."""
    p = subprocess.Popen([sys.executable, '-u', '-m', *map(str, args)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    if p.wait() != 0:
        raise RuntimeError(f'Lệnh lỗi (mã {{p.returncode}}). Dừng lại và gửi thông báo lỗi ở trên cho Claude.')
''')

md('## 2. Phần A: tải và chuẩn bị dữ liệu (runtime CPU)')
code(f'''
import subprocess
from src.weeds.data import build_npz

ZIP = DATA_DRIVE / 'weedsgalore-dataset.zip'
if not ZIP.exists() or ZIP.stat().st_size != {ZIP_BYTES}:
    subprocess.run(['curl', '-L', '--retry', '3', '-o', str(ZIP), '{ZIP_URL}'], check=True)
assert ZIP.stat().st_size == {ZIP_BYTES}, f'Kích thước file zip không đúng: {{ZIP.stat().st_size}}'

if all((DATA_DRIVE / f'weedsgalore_{{s}}.npz').exists() for s in ('train', 'val', 'test')):
    print('Dữ liệu gọn đã có trên Drive, bỏ qua.')
else:
    build_npz(ZIP, DATA_DRIVE)
''')

md('''
## 3. Phần B: kiểm tra GPU và chép dữ liệu về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code('''
import shutil, torch
!nvidia-smi -L
print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang T4 rồi chạy lại mục 1.'
DATA = Path('/content/data')
DATA.mkdir(exist_ok=True)
for s in ('train', 'val', 'test'):
    src = DATA_DRIVE / f'weedsgalore_{s}.npz'
    assert src.exists(), f'Chưa có {src}: chạy Phần A (mục 2) trước.'
    if not (DATA / src.name).exists():
        shutil.copy(src, DATA)
print(sorted(p.name for p in DATA.iterdir()))
''')

md('''
## 4. Chạy thử (khoảng vài phút)

Chạy 1 epoch chỉ 2 vòng lặp, rồi đánh giá trên 2 ô ảnh. Mục đích là bắt lỗi và kiểm tra bộ nhớ GPU (dòng `peak GPU`) ở **cùng batch 8** với lần huấn luyện thật. Bộ nhớ ước tính khoảng 9–10 GB, T4 có khoảng 15 GB. Kết quả chạy thử để ở `/content/smoke`, không dùng để báo cáo. Nếu báo lỗi hết bộ nhớ (out of memory), dừng lại và báo cho Claude.
''')
code('''
shutil.rmtree('/content/smoke', ignore_errors=True)
run('src.weeds.train', '--data', DATA, '--out', '/content/smoke', '--smoke', '--batch', BATCH)
''')

md('''
## 5. Huấn luyện đầy đủ

Tối đa 200 epoch, dừng sớm nếu mIoU trên val không tăng sau 30 epoch, tự dừng trước 3,5 giờ. Mỗi epoch in ra một dòng gồm loss, mIoU val, thời gian và bộ nhớ GPU. Nếu thấy dòng `time budget ... reached`, ở phiên sau chạy lại mục 1, 3 rồi mục này.
''')
code('''
run('src.weeds.train', '--stage', 'train', '--data', DATA, '--out', RUN, '--batch', BATCH)
''')

md('## 6. Đánh giá trên tập test và lưu xác suất dự đoán (chỉ chạy khi huấn luyện đã xong)')
code('''
done = RUN / 'train_done.json'
if done.exists():
    print(done.read_text())
    run('src.weeds.train', '--stage', 'test', '--data', DATA, '--out', RUN)
else:
    print('Huấn luyện chưa xong: ở phiên sau chạy lại mục 1, 3 rồi 5.')
''')

md('## 7. Tầng quyết định và kết luận đạt/không đạt')
code('''
if (RUN / 'probs_test.npz').exists():
    run('src.weeds.analyze', '--data', DATA, '--out', RUN)
else:
    print('Chưa có xác suất dự đoán: chạy mục 6 trước.')
''')

md('## 8. Đóng gói kết quả (không kèm checkpoint mô hình)')
code('''
import zipfile
ZIP_OUT = BASE / f'ket_qua_weeds_pilot_b{BATCH}.zip'
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(RUN.iterdir()):
        if f.suffix in ('.json', '.csv', '.npz'):
            z.write(f, RESULTS / f.name)
print('Đã tạo', ZIP_OUT, f'({ZIP_OUT.stat().st_size / 1e6:.1f} MB)')
''')

md('''
**Bước tiếp theo:** tải `ket_qua_weeds_pilot_b8.zip` về, giải nén vào thư mục gốc của repo (sẽ tạo thư mục `results/weeds_pilot_b8/`, không đụng tới kết quả lần 1 trong `results/weeds_pilot/`), rồi nhờ Claude phân tích.
''')

nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
for i, c in enumerate(nb['cells']):
    c['id'] = f'cell-{i:02d}'
out = ROOT / 'notebooks' / 'colab_weeds_pilot.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('wrote', out, '| embedded modules:', ', '.join(MODULES))
