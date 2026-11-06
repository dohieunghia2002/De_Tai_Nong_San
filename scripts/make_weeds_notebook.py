"""Generate the Colab notebooks with the src/weeds modules embedded, so that they run on Colab without pushing code
to GitHub. Rerun after any change in src/weeds:  python scripts/make_weeds_notebook.py
  notebooks/colab_weeds_pilot.ipynb    train DeepLabv3+, evaluate, decision layer (pilot runs)
  notebooks/colab_weeds_authors.ipynb  evaluate the authors' released checkpoints, decision layer on their model
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['src/__init__.py'] + [f'src/weeds/{m}.py' for m in ('__init__', 'data', 'model', 'train', 'decision',
                                                                 'analyze', 'authors')]
ZIP_URL = 'https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip'
ZIP_BYTES = 336989804  # size served by GFZ on 2026-10-29

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n').splitlines(keepends=True)})


def write_notebook(name):
    """Write the cells collected so far as notebooks/<name> and start a new notebook."""
    nb = {'cells': list(cells), 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                             'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                             'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for i, c in enumerate(nb['cells']):
        c['id'] = f'cell-{i:02d}'
    out = ROOT / 'notebooks' / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('wrote', out, '| embedded modules:', ', '.join(MODULES))
    cells.clear()


files = {m: (ROOT / m).read_text(encoding='utf-8') for m in MODULES}


def setup_code(run_lines, run_print):
    """Section 1 of every notebook: mount Drive, write the embedded sources, define run()."""
    return f'''
import os, sys, json
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/De_Tai_Nong_San/weeds_pilot')
DATA_DRIVE = BASE / 'data'
{run_lines.strip()}
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
{run_print.strip()}

import subprocess
def run(*args):
    """Run a module of the pilot, stream its output, and stop 'Run all' if it fails."""
    p = subprocess.Popen([sys.executable, '-u', '-m', *map(str, args)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    if p.wait() != 0:
        raise RuntimeError(f'Lệnh lỗi (mã {{p.returncode}}). Dừng lại và gửi thông báo lỗi ở trên cho Claude.')
'''


DATA_CODE = f'''
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
'''

def gpu_code(splits):
    """Check the GPU and copy the compact data of `splits` from Drive to the Colab disk."""
    return '''
import shutil, torch
!nvidia-smi -L
print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang T4 rồi chạy lại mục 1.'
DATA = Path('/content/data')
DATA.mkdir(exist_ok=True)
for s in ''' + repr(tuple(splits)) + ''':
    src = DATA_DRIVE / f'weedsgalore_{s}.npz'
    assert src.exists(), f'Chưa có {src}: chạy Phần A (mục 2) trước.'
    if not (DATA / src.name).exists():
        shutil.copy(src, DATA)
print(sorted(p.name for p in DATA.iterdir()))
'''

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
code(setup_code('''
BATCH = 8  # run 2: the paper's batch size; run 1 used 2 (authors' code default) and lives in run_dlv3p_msi_seed0
RUN = BASE / f'run_dlv3p_msi_seed0_b{BATCH}'  # a new folder, so run 1's train_done.json cannot skip training
RESULTS = Path(f'results/weeds_pilot_b{BATCH}')  # folder inside the downloaded zip; run 1 is in results/weeds_pilot
''', "print('Batch', BATCH, '| thư mục kết quả:', RUN)"))

md('## 2. Phần A: tải và chuẩn bị dữ liệu (runtime CPU)')
code(DATA_CODE)

md('''
## 3. Phần B: kiểm tra GPU và chép dữ liệu về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code(gpu_code(('train', 'val', 'test')))

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

write_notebook('colab_weeds_pilot.ipynb')

# ---------------------------------------------------------------------------------------------------------------------
# Comparison with the paper: the authors' released checkpoints evaluated with our code
# ---------------------------------------------------------------------------------------------------------------------
md('''
# Đối sánh với bài báo gốc: đánh giá mô hình của tác giả bằng code của đề tài

Tác giả WeedsGalore (Celikkan và cộng sự, WACV 2025) công bố sẵn các mô hình DeepLabv3+ đã huấn luyện (`ckpts.zip`, 887 MB, trên trang dữ liệu). Notebook này nạp 4 mô hình (ảnh 5 dải và RGB; bài toán 3 lớp và 6 lớp) vào **đúng hàm đánh giá đã cho ra kết quả lần chạy 2** (`src.weeds.train.evaluate`), rồi so với Bảng 3 và Bảng 4 của bài báo. Mỗi mô hình được đánh giá hai lần:
- **fp32**, như code của tác giả: nếu ra đúng số của bài báo (ví dụ mIoU 82,90 cho ảnh 5 dải, 3 lớp), hàm đánh giá của đề tài được kiểm chứng;
- **fp16**, như lần chạy 2 của đề tài: cho biết riêng phép tính fp16 làm lệch bao nhiêu. Khi đó phần còn lại của chênh lệch 3,65 điểm nằm ở khâu huấn luyện.

Sau đó chạy tầng quyết định (phun / không phun / chưa quyết) trên xác suất fp16 của mô hình tác giả, cùng cách làm với mô hình đề tài tự huấn luyện, để so sánh hai mô hình.

Không huấn luyện gì. Thời gian: tải 887 MB (vài phút, chỉ lần đầu, lưu trên Drive) và đánh giá vài phút trên GPU T4.

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Muốn tiết kiệm GPU thì chạy mục 1–3 trên runtime CPU trước (tải dữ liệu và mô hình lên Drive), rồi đổi sang T4 và Run all; các mục đã xong sẽ tự bỏ qua. Nếu một bước bị lỗi, gửi thông báo lỗi cho Claude.

Kết quả nằm trong `MyDrive/De_Tai_Nong_San/weeds_pilot/authors_ckpts/eval/`. Mục cuối tạo file `ket_qua_weeds_authors.zip` để tải về.
''')

md('## 1. Kết nối Drive và ghi mã nguồn (chạy ở **mọi** phiên, cả CPU lẫn GPU)')
code(setup_code('''
RUN = BASE / 'authors_ckpts'  # the authors' ckpts.zip and its evaluation, apart from our own training runs
RESULTS = Path('results/weeds_authors')  # folder inside the downloaded zip
''', "print('Thư mục kết quả:', RUN)"))

md('## 2. Phần A: tải và chuẩn bị dữ liệu (runtime CPU; bỏ qua nếu đã có từ notebook thí nghiệm thử)')
code(DATA_CODE)

md('## 3. Phần A: tải mô hình của tác giả lên Drive (runtime CPU, 887 MB, chỉ lần đầu)')
code('''
from src.weeds.authors import CKPT_URL, CKPT_ZIP_BYTES, CKPTS

CKPT_ZIP = RUN / 'ckpts.zip'
if not CKPT_ZIP.exists() or CKPT_ZIP.stat().st_size != CKPT_ZIP_BYTES:
    subprocess.run(['curl', '-L', '--retry', '3', '-o', str(CKPT_ZIP), CKPT_URL], check=True)
assert CKPT_ZIP.stat().st_size == CKPT_ZIP_BYTES, f'Kích thước ckpts.zip không đúng: {CKPT_ZIP.stat().st_size}'
print('ckpts.zip đã có trên Drive:', CKPT_ZIP)
''')

md('''
## 4. Phần B: kiểm tra GPU, chép dữ liệu và mô hình về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code(gpu_code(('val', 'test')))
code('''
import zipfile
CKPT_DIR = Path('/content/ckpts')
CKPT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(CKPT_ZIP) as z:
    for name in CKPTS:
        member, target = f'ckpts/{name}.pth', CKPT_DIR / f'{name}.pth'
        # a file left half-written by an interrupted extraction has the wrong size and is extracted again
        if not target.exists() or target.stat().st_size != z.getinfo(member).file_size:
            tmp = target.with_suffix('.tmp')
            tmp.write_bytes(z.read(member))
            tmp.replace(target)
print(sorted(p.name for p in CKPT_DIR.iterdir()))
''')

md('''
## 5. Đánh giá 4 mô hình của tác giả và so với bài báo

Mỗi mô hình in một dòng: mIoU test tính bằng fp32 và fp16, mIoU trong bài báo, chênh lệch fp32 so với bài báo, rồi IoU từng lớp (fp32 / bài báo). Chênh lệch cỡ vài phần trăm điểm là bình thường (khác GPU, khác phiên bản PyTorch).
''')
code('''
EVAL = RUN / 'eval'
run('src.weeds.authors', '--data', DATA, '--ckpts', CKPT_DIR, '--out', EVAL)
''')

md('## 6. Tầng quyết định trên mô hình của tác giả (bài toán 3 lớp)')
code('''
for name in ('dlv3p_msi_3', 'dlv3p_rgb_3'):
    print(f'\\n========== {name} ==========')
    run('src.weeds.analyze', '--data', DATA, '--out', EVAL / name)
''')

md('## 7. Đóng gói kết quả (không kèm mô hình; khoảng 100 MB vì có xác suất dự đoán của 2 mô hình 3 lớp)')
code('''
ZIP_OUT = BASE / 'ket_qua_weeds_authors.zip'
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(EVAL.rglob('*')):
        if f.suffix in ('.json', '.csv', '.npz'):
            z.write(f, RESULTS / f.relative_to(EVAL))
print('Đã tạo', ZIP_OUT, f'({ZIP_OUT.stat().st_size / 1e6:.1f} MB)')
''')

md('''
**Bước tiếp theo:** tải `ket_qua_weeds_authors.zip` về, giải nén vào thư mục gốc của repo (sẽ tạo thư mục `results/weeds_authors/`), rồi nhờ Claude phân tích.
''')

write_notebook('colab_weeds_authors.ipynb')
