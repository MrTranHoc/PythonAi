# DeeplabV3+ Cityscapes

Dự án huấn luyện mô hình **DeepLabV3+** cho bài toán phân đoạn ảnh semantic segmentation trên tập dữ liệu **Cityscapes**.

## Tính năng chính
- Huấn luyện DeepLabV3+ với backbone ResNet (mặc định `resnet50`).
- Chuẩn hóa pipeline cho Cityscapes (`leftImg8bit` + `gtFine`).
- Hỗ trợ mixed precision (AMP), checkpoint, và đánh giá mIoU.
- Có script inference để dự đoán và xuất ảnh mask màu.

## Cấu trúc thư mục
```text
DeeplabV3_CityScapes/
├── configs/
│   └── cityscapes_resnet50.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── src/
│   ├── cityscapes_labels.py
│   ├── data.py
│   ├── model.py
│   ├── train_utils.py
│   ├── engine.py
│   └── metrics.py
├── checkpoints/
├── notebooks/
├── requirements.txt
└── README.md
```

## 1) Cài đặt
```bash
cd DeeplabV3_CityScapes
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Chuẩn bị dữ liệu Cityscapes
Tải dữ liệu từ website Cityscapes và đảm bảo cấu trúc:

```text
/path/to/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

## 3) Huấn luyện
```bash
python scripts/train.py --config configs/cityscapes_resnet50.yaml --data-root /path/to/cityscapes
```

## 4) Đánh giá
```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --data-root /path/to/cityscapes
```

## 5) Inference
```bash
python scripts/infer.py \
  --checkpoint checkpoints/best.pt \
  --input /path/to/image.png \
  --output outputs/prediction.png
```

## Ghi chú
- Cityscapes có 19 lớp train IDs, mô hình được cấu hình `num_classes=19`.
- Với GPU nhỏ, giảm `batch_size` hoặc `image_size` trong file config.
