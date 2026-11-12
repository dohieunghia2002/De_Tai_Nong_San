"""Progress demo of the spraying decision support system, for showing the professor (Streamlit, runs locally).

  .venv/Scripts/streamlit run demo/app.py        (from the repo root; then open http://localhost:8501)

Needs the dataset zip (data/weedsgalore/) and the unpacked Colab results (results/, including probs_*.npz and
xai_maps.npz, which are not in git). Nothing is trained or rescored by a network here: pages read results/ and
recompute decisions from the saved probability maps with src.weeds.
"""
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import streamlit as st  # noqa: E402

import core  # noqa: E402,F401  (puts the repo root and scripts/ on sys.path for the pages)

st.set_page_config(page_title='Demo tiến độ: hỗ trợ ra quyết định phun thuốc', page_icon=':material/grass:',
                   layout='wide')

pages = {
    'Chạy thử hệ thống': [
        st.Page('views/pipeline.py', title='Ảnh → tìm cỏ → quyết định → giải thích', icon=':material/play_circle:',
                default=True),
    ],
    'Kết quả đánh giá trên tập kiểm tra': [
        st.Page('views/overview.py', title='Tổng quan kết quả', icon=':material/home:', url_path='tong-quan'),
        st.Page('views/dataset.py', title='1. Dữ liệu ảnh drone', icon=':material/photo_library:', url_path='du-lieu'),
        st.Page('views/segmentation.py', title='2. Tìm cỏ trên ảnh', icon=':material/image_search:', url_path='tim-co'),
        st.Page('views/decision.py', title='3. Quyết định phun', icon=':material/grid_on:', url_path='quyet-dinh'),
        st.Page('views/explanation.py', title='4. Giải thích từng ô', icon=':material/lightbulb:',
                url_path='giai-thich'),
        st.Page('views/faithfulness.py', title='5. Kiểm chứng lời giải thích', icon=':material/fact_check:',
                url_path='kiem-chung'),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption('Hệ thống hỗ trợ ra quyết định phun thuốc trừ cỏ từ ảnh drone đa phổ. "Chạy thử" chạy mô hình thật trên '
               'CPU của máy này; các trang "Kết quả đánh giá" đọc số liệu từ `results/` hoặc tính lại từ bản đồ xác suất '
               'đã lưu.')
nav.run()
