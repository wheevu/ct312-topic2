# Ứng dụng trực quan hóa

Ứng dụng Streamlit này dùng để xem nhanh dữ liệu, kết quả gom nhóm và các hình minh họa của Chủ đề 2.

## Cài đặt

```bash
python -m pip install -r app/requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app/streamlit_app.py
```

Nếu thiếu dữ liệu hoặc kết quả, tạo lại bộ dữ liệu trước:

```bash
python scripts/topic2/01_prepare_demo_data.py
python scripts/topic2/02_validate_handoff.py
```
