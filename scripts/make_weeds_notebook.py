"""Generate the Colab notebooks with the src/weeds modules embedded, so that they run on Colab without pushing code
to GitHub. Rerun after any change in src/weeds:  python scripts/make_weeds_notebook.py
  notebooks/colab_weeds_pilot.ipynb    train DeepLabv3+, evaluate, decision layer (pilot runs)
  notebooks/colab_weeds_authors.ipynb  evaluate the authors' released checkpoints, decision layer on their model
  notebooks/colab_weeds_runs.ipynb     further training runs (seeds, schedules) of src/weeds/runs.py
  notebooks/colab_weeds_xai.ipynb      explanations of the spraying decisions and their faithfulness (src/weeds/explain.py)
  notebooks/colab_weeds_cfcheck.ipynb  model check of the counterfactual explanations (src/weeds/counterfactual_check.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['src/__init__.py'] + [f'src/weeds/{m}.py' for m in ('__init__', 'data', 'model', 'train', 'decision',
                                                                 'analyze', 'authors', 'runs', 'explain',
                                                                 'counterfactual', 'counterfactual_check')]
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

# ---------------------------------------------------------------------------------------------------------------------
# Further training runs: seeds and schedules (src/weeds/runs.py), to explain the gap to the paper
# ---------------------------------------------------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from src.weeds.runs import RUNS, fixed  # noqa: E402

SECONDS_PER_EPOCH = 17.5  # run 2 on a Colab T4, batch 8


def label(r):
    """Vietnamese description of a run's schedule, built from its own fields."""
    s = (f"{r['epochs']} epoch cố định" if fixed(r)
         else f"dừng sớm sau {r['patience']} epoch không cải thiện (tối đa {r['epochs']} epoch)")
    return s + (', learning rate giảm dần (cosine)' if r['lr_schedule'] == 'cosine' else ', learning rate không đổi')


def duration(r):
    minutes = round(r['epochs'] * SECONDS_PER_EPOCH / 60)
    return f'khoảng {minutes} phút' if fixed(r) else f'tối đa khoảng {minutes} phút'


run_table = '\n'.join(f"| {i + 1} | `{r['name']}` | {label(r)} | {r['seed']} | {duration(r)} |"
                      for i, r in enumerate(RUNS))
md(f'''
# Huấn luyện thêm: nhiều seed và biến thể, để thu hẹp khoảng cách với bài báo

Lần chạy 2 của đề tài đạt mIoU test 79,25 (1 seed), thấp hơn bài báo 3,65 điểm. Mô hình công bố của tác giả, chạy qua cùng code đánh giá, ra đúng 82,90, nên chênh lệch nằm ở khâu huấn luyện. Code huấn luyện của tác giả chỉ khác code đề tài ở lịch huấn luyện: 100 epoch cố định, không dừng sớm. Notebook này chạy các lần sau theo thứ tự ưu tiên, đều với batch 8 và ảnh 5 dải:

| Thứ tự | Tên | Lịch huấn luyện | Seed | Thời gian ước tính |
| --- | --- | --- | --- | --- |
{run_table}

Mỗi lần chạy: huấn luyện, rồi đánh giá trên tập test và chạy tầng quyết định cho **hai mô hình**: mô hình có mIoU val cao nhất (như lần chạy 2) và mô hình ở epoch cuối. Lý do: lần chạy 2 có val 83,94 nhưng test chỉ 79,25, còn mô hình của tác giả có val 79,53 và test 82,90; chọn epoch có val cao nhất trong nhiều epoch dao động có thể chọn trúng một epoch "may mắn" trên val. Mô hình epoch cuối của lần chạy 2 cũng được đánh giá ở đầu phiên đầu tiên (vài phút).

Tổng cộng tối đa khoảng 4–4,5 giờ GPU, tức **2 phiên Colab**.

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Mỗi phiên tự dừng huấn luyện sau khoảng 3,5 giờ tính từ lúc máy Colab khởi động (chạy lại notebook trên cùng máy không làm đồng hồ đếm lại). Khi thấy dòng "Hết thời gian của phiên này", hôm sau mở lại notebook và **Run all lại**: lần đã xong tự bỏ qua, lần đang dở chạy tiếp từ checkpoint. Cuối notebook có lệnh ghi hết dữ liệu lên Drive; chờ nó xong rồi mới đóng phiên.

- Mỗi lần chạy chiếm khoảng 0,25 GB trên Drive sau khi xong (mô hình tốt nhất và xác suất dự đoán của hai mô hình); trong lúc chạy cần thêm khoảng 0,5 GB cho checkpoint, được xóa khi lần chạy xong. File xóa trên Drive nằm trong Thùng rác 30 ngày: nếu Drive gần đầy, hãy dọn Thùng rác.
- Giữ tab Colab mở trong lúc chạy. Nếu một bước bị lỗi, gửi thông báo lỗi cho Claude.

Kết quả nằm trong `MyDrive/De_Tai_Nong_San/weeds_pilot/runs/`. Mục cuối tạo file `ket_qua_weeds_runs.zip` để tải về.
''')

md('## 1. Kết nối Drive và ghi mã nguồn (chạy ở **mọi** phiên)')
code(setup_code('''
RUN = BASE / 'runs'  # one folder per run of src/weeds/runs.py; run 2 stays in run_dlv3p_msi_seed0_b8
RESULTS = Path('results/weeds_runs')  # folder inside the downloaded zip
''', "print('Thư mục kết quả:', RUN)"))

md('## 2. Phần A: tải và chuẩn bị dữ liệu (bỏ qua nếu đã có từ các notebook trước)')
code(DATA_CODE)

md('''
## 3. Phần B: kiểm tra GPU và chép dữ liệu về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code(gpu_code(('train', 'val', 'test')))
code('''
import time
from src.weeds.runs import RUNS, by_schedule, summarize, train_args

# the session clock counts from the start of this Colab VM, so running the notebook again on it does not reset it
T0 = time.time() - float(Path('/proc/uptime').read_text().split()[0])
SESSION_HOURS = 3.5  # training stops in time to test, analyse, package and sync to Drive before the Colab limit
PILOT = BASE / 'run_dlv3p_msi_seed0_b8'  # run 2
print(f'Máy Colab này đã chạy {(time.time() - T0) / 3600:.2f} giờ; huấn luyện dừng khi đủ {SESSION_HOURS} giờ.')
''')

md('''
## 4. Chạy thử (chỉ ở phiên đầu, vài phút)

Chạy 1 epoch chỉ 2 vòng lặp với learning rate giảm dần, rồi đánh giá cả mô hình tốt nhất lẫn mô hình epoch cuối, để bắt lỗi trước khi huấn luyện thật. Từ phiên thứ hai trở đi, mục này tự bỏ qua.
''')
code('''
if not any((RUN / r['name'] / 'history.json').exists() for r in RUNS):
    shutil.rmtree('/content/smoke', ignore_errors=True)
    run('src.weeds.train', '--data', DATA, '--out', '/content/smoke', '--smoke', '--batch', 8, '--lr-schedule', 'cosine')
    run('src.weeds.train', '--stage', 'test', '--weights', 'last', '--smoke', '--data', DATA, '--out', '/content/smoke')
else:
    print('Đã có lần chạy bắt đầu từ phiên trước: bỏ qua chạy thử.')
''')

md('''
## 5. Huấn luyện, đánh giá và tầng quyết định cho từng lần chạy

Mỗi epoch in một dòng như notebook thí nghiệm thử. Lần đã xong được bỏ qua. Khi một lần huấn luyện xong: đánh giá mô hình tốt nhất (`best.pt`, giữ lại) và mô hình epoch cuối (kết quả trong thư mục con `last/`), rồi xóa checkpoint `last.pt` (khoảng 0,5 GB).
''')
code('''
def evaluate_models(out):
    """Test and decision layer for the model chosen on validation and for the final-epoch model (each step once)."""
    if not (out / 'go_no_go.json').exists():
        run('src.weeds.train', '--stage', 'test', '--data', DATA, '--out', out)
        run('src.weeds.analyze', '--data', DATA, '--out', out)
    if (out / 'last.pt').exists() and not (out / 'last' / 'go_no_go.json').exists():
        run('src.weeds.train', '--stage', 'test', '--weights', 'last', '--data', DATA, '--out', out)
        run('src.weeds.analyze', '--data', DATA, '--out', out / 'last')

# run 2: only its final-epoch model is new (its checkpoint is kept: it belongs to the earlier notebook)
if (PILOT / 'last.pt').exists() and not (PILOT / 'last' / 'go_no_go.json').exists():
    print('========== lần chạy 2: mô hình epoch cuối ==========')
    run('src.weeds.train', '--stage', 'test', '--weights', 'last', '--data', DATA, '--out', PILOT)
    run('src.weeds.analyze', '--data', DATA, '--out', PILOT / 'last')

for r in RUNS:
    out = RUN / r['name']
    if not (out / 'train_done.json').exists():
        left = SESSION_HOURS - (time.time() - T0) / 3600
        if left < 0.3:
            print(f'Hết thời gian của phiên này (còn {left:.2f} giờ). Phiên sau: Run all lại, các lần đã xong tự bỏ qua.')
            break
        print()
        print(f"========== {r['name']} | phiên còn {left:.2f} giờ ==========")
        run('src.weeds.train', '--stage', 'train', '--data', DATA, '--out', out, *train_args(r),
            '--budget-hours', f'{left - 0.15:.2f}')  # 0.15 h kept for the two evaluations and packaging
        if not (out / 'train_done.json').exists():
            print(f"{r['name']}: chưa huấn luyện xong, sẽ chạy tiếp từ checkpoint ở phiên sau.")
            break
    evaluate_models(out)  # also completes a run whose evaluation an earlier session left unfinished
    for f in ('last.pt', 'last.tmp'):
        (out / f).unlink(missing_ok=True)
    print(f"{r['name']}: xong.")
''')

md('## 6. Tổng hợp các lần chạy đã xong (kèm lần chạy 2)')
code('''
import pandas as pd
pd.set_option('display.width', 220)
dirs = {'es_seed0 (lần chạy 2)': PILOT, **{r['name']: RUN / r['name'] for r in RUNS}}
df = summarize(dirs)
if df.empty:
    print('Chưa có lần chạy nào xong.')
else:
    print(df.round(3).to_string(index=False))
    print()
    print('mIoU test theo lịch huấn luyện và cách chọn mô hình (bài báo và mô hình công bố của tác giả: 82,90):')
    print(by_schedule(df).round(2).to_string(index=False))
    df.to_csv(RUN / 'runs_summary.csv', index=False)
if df.attrs['skipped']:
    print('Chưa có kết quả:', ', '.join(df.attrs['skipped']))
''')

md('## 7. Đóng gói kết quả (chỉ file json và csv; mô hình và xác suất dự đoán ở lại trên Drive)')
code('''
import zipfile
ZIP_OUT = BASE / 'ket_qua_weeds_runs.zip'
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(RUN.rglob('*')):
        if f.suffix in ('.json', '.csv'):
            z.write(f, RESULTS / f.relative_to(RUN))
    for f in sorted((PILOT / 'last').glob('*')):  # run 2's final-epoch model
        if f.suffix in ('.json', '.csv'):
            z.write(f, RESULTS / 'pilot_b8_seed0' / 'last' / f.name)
print('Đã tạo', ZIP_OUT, f'({ZIP_OUT.stat().st_size / 1e6:.1f} MB)')
''')

md('## 8. Ghi hết dữ liệu lên Drive (chờ dòng "Đã ghi hết" rồi mới đóng phiên)')
code('''
drive.flush_and_unmount()  # checkpoints of 0.5 GB may still be syncing; the next session resumes from them
print('Đã ghi hết dữ liệu lên Drive. Có thể đóng phiên Colab.')
''')

md('''
**Bước tiếp theo:** khi cả 6 lần đã xong (hoặc muốn xem kết quả giữa chừng), tải `ket_qua_weeds_runs.zip` về, giải nén vào thư mục gốc của repo (sẽ tạo thư mục `results/weeds_runs/`), rồi nhờ Claude phân tích.
''')

write_notebook('colab_weeds_runs.ipynb')

# ---------------------------------------------------------------------------------------------------------------------
# Explanations (XAI) of the spraying decisions and their faithfulness (src/weeds/explain.py)
# ---------------------------------------------------------------------------------------------------------------------
# decisions and thresholds reported for the authors' 5-band model, so that the notebook checks it explains those
EXPECTED_FILE = ROOT / 'results' / 'weeds_authors' / 'dlv3p_msi_3' / 'go_no_go.json'
EXPECTED = EXPECTED_FILE.read_text(encoding='utf-8') if EXPECTED_FILE.exists() else None

md('''
# Giải thích quyết định phun thuốc (XAI) và kiểm chứng độ trung thực của lời giải thích

Mô hình được giải thích: mô hình công bố của tác giả (DeepLabv3+, ảnh 5 dải, 3 lớp; mIoU test 82,90 như bài báo), là mô hình phân vùng chính của đề tài. Các ô được giải thích: 216 ô lưới 0,5 m của 24 ô ảnh test thuộc 3 đợt bay trước khi phun thuốc, với quyết định phun / không phun / chưa chắc ở thiết lập chính.

Quyết định của một ô chỉ phụ thuộc vào điểm số s = trung bình xác suất cỏ trong ô, nên giải thích quyết định là chỉ ra **điểm ảnh nào và dải phổ nào** làm s cao lên:
- **Integrated Gradients**, so với ảnh "đất trống" (mỗi dải lấy giá trị trung vị của các điểm ảnh nền đất trong tập train): cho mức đóng góp của từng điểm ảnh và từng dải; tổng đóng góp phải bằng s(ảnh) − s(đất trống).
- **Grad-CAM** trên đặc trưng của bộ giải mã.
- **Bản đồ xác suất cỏ** của chính mô hình, làm mốc so sánh.
- **Mức quan trọng của từng dải phổ:** thay một dải bằng giá trị đất trên cả ô ảnh, xem s giảm bao nhiêu.

**Kiểm chứng độ trung thực** (với các ô PHUN và CHƯA CHẮC): thay các vùng 5 cm × 5 cm mà lời giải thích xếp cao nhất bằng đất (0,5% đến 10% số vùng của ô ảnh), đo s giảm bao nhiêu và quyết định PHUN có đổi không. Đối chứng: thay cùng số vùng **chọn ngẫu nhiên** trên cả ô ảnh, và ngẫu nhiên **ngay trong ô lưới** (10 lần mỗi loại). Lời giải thích trung thực phải làm s giảm mạnh hơn đối chứng ngẫu nhiên.

Quyết định của các ô được tính một lần ở phiên đầu và đối chiếu với kết quả đã báo cáo cho mô hình này (`results/weeds_authors`); nếu lệch, notebook in dòng `WARNING`.

Thời gian ước tính: khoảng 30–45 phút GPU (mục 5 in dòng `estimated full run` sau khi chạy thử 1 ô ảnh). Kết quả lưu theo từng ô ảnh, nên nếu bị ngắt thì Run all lại sẽ chạy tiếp.

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Mục 8 tạo `ket_qua_weeds_xai.zip` (khoảng 30–40 MB, có bản đồ giải thích thu nhỏ) khi đã chạy xong.
''')

md('## 1. Kết nối Drive và ghi mã nguồn (chạy ở **mọi** phiên)')
code(setup_code('''
RUN = BASE / 'xai_authors_msi3'  # explanations of the authors' 5-band, 3-class model
RESULTS = Path('results/weeds_xai')  # folder inside the downloaded zip
CKPT_ZIP = BASE / 'authors_ckpts' / 'ckpts.zip'  # shared with colab_weeds_authors.ipynb
''', "print('Thư mục kết quả:', RUN)"))

md('## 2. Phần A: tải và chuẩn bị dữ liệu (bỏ qua nếu đã có)')
code(DATA_CODE)

md('## 3. Phần A: tải mô hình của tác giả lên Drive (887 MB, bỏ qua nếu đã có)')
code('''
from src.weeds.authors import CKPT_URL, CKPT_ZIP_BYTES

CKPT_ZIP.parent.mkdir(parents=True, exist_ok=True)
if not CKPT_ZIP.exists() or CKPT_ZIP.stat().st_size != CKPT_ZIP_BYTES:
    subprocess.run(['curl', '-L', '--retry', '3', '-o', str(CKPT_ZIP), CKPT_URL], check=True)
assert CKPT_ZIP.stat().st_size == CKPT_ZIP_BYTES, f'Kích thước ckpts.zip không đúng: {CKPT_ZIP.stat().st_size}'
print('ckpts.zip đã có trên Drive:', CKPT_ZIP)
''')

md('''
## 4. Phần B: kiểm tra GPU, chép dữ liệu và mô hình về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code(gpu_code(('train', 'test')))
code('''
import time, zipfile
CKPT = Path('/content/ckpts/dlv3p_msi_3.pth')
CKPT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(CKPT_ZIP) as z:
    member = 'ckpts/dlv3p_msi_3.pth'
    if not CKPT.exists() or CKPT.stat().st_size != z.getinfo(member).file_size:
        tmp = CKPT.with_suffix('.tmp')
        tmp.write_bytes(z.read(member))
        tmp.replace(CKPT)
# the session clock counts from the start of this Colab VM, so running the notebook again on it does not reset it
T0 = time.time() - float(Path('/proc/uptime').read_text().split()[0])
SESSION_HOURS = 3.5
print(CKPT, f'{CKPT.stat().st_size / 1e6:.0f} MB | máy Colab đã chạy {(time.time() - T0) / 3600:.2f} giờ')
''' + (f'''
EXPECT = Path('/content/expected_go_no_go.json')  # what results/weeds_authors reported for this model
EXPECT.write_text({EXPECTED!r})
EXPECT_ARGS = ['--expect', EXPECT]
''' if EXPECTED else '''
EXPECT_ARGS = []  # no reported results were available when this notebook was generated
'''))

md('''
## 5. Chạy thử trên 1 ô ảnh (vài phút, chỉ ở phiên đầu)

Bắt lỗi và đo thời gian cho 1 ô ảnh với thiết lập đầy đủ. Dòng `estimated full run` cho biết thời gian ước tính của mục 6; dòng `completeness` phải có sai số nhỏ (cỡ 1e-4 trở xuống). Nếu báo lỗi hết bộ nhớ (out of memory) hoặc có dòng `WARNING`, dừng lại và báo cho Claude.
''')
code('''
if (RUN / 'parts').exists() and any((RUN / 'parts').iterdir()):
    print('Đã có kết quả từ phiên trước: bỏ qua chạy thử.')
else:
    shutil.rmtree('/content/xai_smoke', ignore_errors=True)
    run('src.weeds.explain', '--data', DATA, '--ckpt', CKPT, '--out', '/content/xai_smoke', '--limit-tiles', 1,
        *EXPECT_ARGS)
''')

md('## 6. Giải thích toàn bộ 216 ô lưới và kiểm chứng độ trung thực (chạy tiếp được nếu bị ngắt)')
code('''
left = SESSION_HOURS - (time.time() - T0) / 3600
if left < 0.3:
    print(f'Phiên này gần hết giờ (còn {left:.2f} giờ): mở phiên mới rồi Run all lại.')
else:
    run('src.weeds.explain', '--data', DATA, '--ckpt', CKPT, '--out', RUN, '--budget-hours', f'{left - 0.2:.2f}',
        *EXPECT_ARGS)
if not (RUN / 'xai_summary.json').exists():
    print('Chưa xong: ở phiên sau Run all lại, các ô ảnh đã xong được giữ lại.')
''')

md('''
## 7. Kết quả kiểm chứng độ trung thực

`mean_rel_drop`: s giảm trung bình bao nhiêu phần (so với s ban đầu) khi xóa các vùng được xếp cao nhất. `beats_random_*`: tỉ lệ ô mà lời giải thích làm s giảm nhiều hơn đối chứng ngẫu nhiên. `spray_flip_rate`: tỉ lệ ô PHUN không còn là PHUN sau khi xóa.
''')
code('''
import pandas as pd
summary = json.loads((RUN / 'xai_summary.json').read_text()) if (RUN / 'xai_summary.json').exists() else None
if summary:
    check = summary['decision_check'] or {}
    print('Quyết định:', summary['decisions'], '| lệch so với kết quả đã báo cáo:', check.get('max_abs_diff'),
          '| ô đổi quyết định khi chấm lại:', summary['rescored_decision_differs'])
    print('Sai số tổng Integrated Gradients:', summary['ig_completeness_abs_error'])
    rows = [{'method': m, 'fraction': float(f), **v} for m, by_f in summary['faithfulness'].items() for f, v in by_f.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
''')

md('## 8. Đóng gói kết quả (chỉ khi đã chạy xong)')
code('''
ZIP_OUT = BASE / 'ket_qua_weeds_xai.zip'
if (RUN / 'xai_summary.json').exists():
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(RUN.glob('*')):
            if f.suffix in ('.json', '.csv', '.npz'):
                z.write(f, RESULTS / f.name)
    print('Đã tạo', ZIP_OUT, f'({ZIP_OUT.stat().st_size / 1e6:.1f} MB)')
else:
    print('Chưa chạy xong nên chưa đóng gói: ở phiên sau Run all lại.')
''')

md('## 9. Ghi hết dữ liệu lên Drive (chờ dòng "Đã ghi hết" rồi mới đóng phiên)')
code('''
drive.flush_and_unmount()
print('Đã ghi hết dữ liệu lên Drive. Có thể đóng phiên Colab.')
''')

md('''
**Bước tiếp theo:** tải `ket_qua_weeds_xai.zip` về, giải nén vào thư mục gốc của repo (sẽ tạo thư mục `results/weeds_xai/`), rồi nhờ Claude phân tích.
''')

write_notebook('colab_weeds_xai.ipynb')

# ---------------------------------------------------------------------------------------------------------------------
# Model check of the counterfactual explanations (src/weeds/counterfactual_check.py)
# ---------------------------------------------------------------------------------------------------------------------
# the decisions, thresholds and bare-soil values of the XAI run, so that the check is on exactly those decisions
XAI_DIR = ROOT / 'results' / 'weeds_xai'
CF_INPUTS = {name: (XAI_DIR / name).read_text(encoding='utf-8') if (XAI_DIR / name).exists() else None
             for name in ('decisions.csv', 'soil_baseline.json')}

md('''
# Kiểm chứng giải thích phản thực bằng mô hình

Giải thích phản thực (`src/weeds/counterfactual.py`, chạy trên máy) trả lời câu hỏi: *bỏ ít nhất bao nhiêu vùng cỏ 5 cm × 5 cm thì ô lưới hết PHUN, hoặc thành KHÔNG PHUN?* Câu trả lời đó là **ước lượng**: lấy điểm số của ô trừ đi phần xác suất cỏ của các vùng bị bỏ, không chạy lại mô hình. Đối chiếu ở 5 mức che cho thấy ước lượng khớp mô hình khoảng 95% nhưng hơi lạc quan ở các ô sát ngưỡng.

Notebook này cho **mô hình chấm lại thật**: với mỗi ô PHUN và CHƯA CHẮC (113 ô, mô hình công bố của tác giả, đúng các quyết định đã phân tích trong `results/weeds_xai`), lần lượt thay 1, 2, 3, ... vùng nhiều cỏ nhất bằng đất rồi cho mô hình chấm lại ô ảnh, cho đến khi mô hình ra KHÔNG PHUN và đã vượt cả hai số ước lượng thêm 2 vùng (hoặc đã bỏ hết các vùng của ô). Kết quả cho biết, ở từng ô, bỏ đúng các vùng mà lời giải thích chỉ ra thì mô hình đổi quyết định ở vùng thứ mấy, và ước lượng lệch bao nhiêu vùng. Đây là kiểm chứng câu giải thích "bỏ k vùng này thì ô hết phun"; nó không chứng minh k là ít nhất với mô hình, vì một tập vùng khác có thể làm đổi quyết định với ít vùng hơn.

Khối lượng: khoảng 2.200 lần chạy mô hình nếu mô hình khớp ước lượng (bằng khoảng 1/4 phép che trong notebook XAI); mục 5 in thời gian ước tính sau khi chạy thử 1 ô ảnh. Kết quả lưu theo từng ô ảnh, nên nếu bị ngắt thì Run all lại sẽ chạy tiếp. Chỉ cần tập test (không cần tập train).

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*. Mục 8 tạo `ket_qua_weeds_cfcheck.zip` (chỉ có csv và json) khi đã chạy xong.
''')

md('## 1. Kết nối Drive và ghi mã nguồn (chạy ở **mọi** phiên)')
code(setup_code('''
RUN = BASE / 'cfcheck_authors_msi3'  # model check of the counterfactuals of the authors' 5-band, 3-class model
RESULTS = Path('results/weeds_cfcheck')  # folder inside the downloaded zip
CKPT_ZIP = BASE / 'authors_ckpts' / 'ckpts.zip'  # shared with colab_weeds_authors.ipynb and colab_weeds_xai.ipynb
''', "print('Thư mục kết quả:', RUN)"))

md('## 2. Phần A: tải và chuẩn bị dữ liệu (bỏ qua nếu đã có)')
code(DATA_CODE)

md('## 3. Phần A: tải mô hình của tác giả lên Drive (887 MB, bỏ qua nếu đã có)')
code('''
from src.weeds.authors import CKPT_URL, CKPT_ZIP_BYTES

CKPT_ZIP.parent.mkdir(parents=True, exist_ok=True)
if not CKPT_ZIP.exists() or CKPT_ZIP.stat().st_size != CKPT_ZIP_BYTES:
    subprocess.run(['curl', '-L', '--retry', '3', '-o', str(CKPT_ZIP), CKPT_URL], check=True)
assert CKPT_ZIP.stat().st_size == CKPT_ZIP_BYTES, f'Kích thước ckpts.zip không đúng: {CKPT_ZIP.stat().st_size}'
print('ckpts.zip đã có trên Drive:', CKPT_ZIP)
''')

md('''
## 4. Phần B: kiểm tra GPU, chép dữ liệu và mô hình về máy Colab

Trước mục này: *Runtime → Change runtime type → T4 GPU*, rồi chạy lại mục 1.
''')
code(gpu_code(('test',)))
code(f'''
import time, zipfile
CKPT = Path('/content/ckpts/dlv3p_msi_3.pth')
CKPT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(CKPT_ZIP) as z:
    member = 'ckpts/dlv3p_msi_3.pth'
    if not CKPT.exists() or CKPT.stat().st_size != z.getinfo(member).file_size:
        tmp = CKPT.with_suffix('.tmp')
        tmp.write_bytes(z.read(member))
        tmp.replace(CKPT)
# the decisions, thresholds and bare-soil values of results/weeds_xai, embedded when this notebook was generated
CF_INPUTS = {CF_INPUTS!r}
assert all(CF_INPUTS.values()), 'Notebook được sinh khi chưa có results/weeds_xai: báo cho Claude.'
INPUTS = Path('/content/cf_inputs')
INPUTS.mkdir(exist_ok=True)
for name, text in CF_INPUTS.items():
    (INPUTS / name).write_text(text)
CF_ARGS = ['--ckpt', CKPT, '--decisions', INPUTS / 'decisions.csv', '--soil', INPUTS / 'soil_baseline.json']
# the session clock counts from the start of this Colab VM, so running the notebook again on it does not reset it
T0 = time.time() - float(Path('/proc/uptime').read_text().split()[0])
SESSION_HOURS = 3.5
print(CKPT, f'{{CKPT.stat().st_size / 1e6:.0f}} MB | máy Colab đã chạy {{(time.time() - T0) / 3600:.2f}} giờ')
''')

md('''
## 5. Chạy thử trên 1 ô ảnh (vài phút, chỉ ở phiên đầu)

Bắt lỗi và đo thời gian. Dòng `estimated full run` cho biết thời gian ước tính của mục 6 (là cận dưới: lâu hơn nếu mô hình cần bỏ nhiều vùng hơn ước lượng). Nếu báo lỗi hết bộ nhớ (out of memory), dừng lại và báo cho Claude.
''')
code('''
if (RUN / 'parts').exists() and any((RUN / 'parts').iterdir()):
    print('Đã có kết quả từ phiên trước: bỏ qua chạy thử.')
else:
    shutil.rmtree('/content/cfcheck_smoke', ignore_errors=True)
    run('src.weeds.counterfactual_check', '--data', DATA, *CF_ARGS, '--out', '/content/cfcheck_smoke',
        '--limit-tiles', 1)
''')

md('## 6. Kiểm chứng toàn bộ 113 ô PHUN và CHƯA CHẮC (chạy tiếp được nếu bị ngắt)')
code('''
left = SESSION_HOURS - (time.time() - T0) / 3600
if left < 0.3:
    print(f'Phiên này gần hết giờ (còn {left:.2f} giờ): mở phiên mới rồi Run all lại.')
else:
    run('src.weeds.counterfactual_check', '--data', DATA, *CF_ARGS, '--out', RUN, '--budget-hours', f'{left - 0.2:.2f}')
if not (RUN / 'cfcheck_summary.json').exists():
    print('Chưa xong: ở phiên sau Run all lại, các ô ảnh đã xong được giữ lại.')
''')

md('''
## 7. Kết quả

`k_not_spray`: bỏ lần lượt các vùng theo lời giải thích thì đến vùng thứ mấy ô hết PHUN; `k_no_spray`: đến vùng thứ mấy ô thành KHÔNG PHUN. `exact`: số ô mà ước lượng đúng bằng mô hình; `estimate_too_small`: số ô mà ước lượng ít vùng hơn mô hình cần (ước lượng lạc quan). `reverts_*`: số ô mà quyết định đổi ngược lại khi bỏ thêm vùng.
''')
code('''
summary = json.loads((RUN / 'cfcheck_summary.json').read_text()) if (RUN / 'cfcheck_summary.json').exists() else None
if summary:
    for key in ('rescored_decision_differs', 'k_not_spray', 'k_not_spray_fragile', 'k_no_spray', 'reverts_to_spray',
                'reverts_from_no_spray', 'curve', 'forward_passes', 'seconds_per_cell'):
        print(key, json.dumps(summary[key], ensure_ascii=False))
''')

md('## 8. Đóng gói kết quả (chỉ khi đã chạy xong)')
code('''
ZIP_OUT = BASE / 'ket_qua_weeds_cfcheck.zip'
if (RUN / 'cfcheck_summary.json').exists():
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(RUN.glob('*')):
            if f.suffix in ('.json', '.csv'):
                z.write(f, RESULTS / f.name)
    print('Đã tạo', ZIP_OUT, f'({ZIP_OUT.stat().st_size / 1e6:.1f} MB)')
else:
    print('Chưa chạy xong nên chưa đóng gói: ở phiên sau Run all lại.')
''')

md('## 9. Ghi hết dữ liệu lên Drive (chờ dòng "Đã ghi hết" rồi mới đóng phiên)')
code('''
drive.flush_and_unmount()
print('Đã ghi hết dữ liệu lên Drive. Có thể đóng phiên Colab.')
''')

md('''
**Bước tiếp theo:** tải `ket_qua_weeds_cfcheck.zip` về thư mục gốc của repo rồi nhờ Claude giải nén (vào `results/weeds_cfcheck/`) và phân tích.
''')

write_notebook('colab_weeds_cfcheck.ipynb')
