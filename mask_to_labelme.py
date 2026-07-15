"""
将已有 PNG 标签批量转成 LabelMe JSON 格式，
让 LabelMe 可以直接加载预标注多边形。
"""
import json, base64
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(r'TP2020数据集/eiseg_ready')
IMG_DIR = ROOT / 'images'
LBL_DIR = ROOT / 'labels'
OUT_DIR = ROOT / 'labels_labelme'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: '湿地', 1: '水体', 2: '湖泊'}
MIN_AREA = 30  # 最小多边形面积，过滤噪点


def mask_to_shapes(label, class_names):
    """将单通道标签图转为 LabelMe shapes 列表"""
    h, w = label.shape
    shapes = []
    for cls_id in [0, 1, 2]:  # 只处理湿地、水体、湖泊
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[label == cls_id] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            approx = cv2.approxPolyDP(cnt, 1.0, True)
            pts = approx.squeeze().tolist()
            if not pts:
                continue
            if isinstance(pts[0], (int, float)):
                pts = [pts]
            shapes.append({
                'label': class_names[cls_id],
                'points': pts,
                'group_id': None,
                'description': '',
                'shape_type': 'polygon',
                'flags': {}
            })
    return shapes


def main():
    img_files = sorted(IMG_DIR.glob('*.png'))
    print(f'共 {len(img_files)} 张图片，开始转换...')

    for img_path in tqdm(img_files, desc='转换中'):
        name = img_path.stem

        # 读取标签
        lbl_path = LBL_DIR / f'{name}.png'
        if not lbl_path.exists():
            continue
        raw = np.fromfile(str(lbl_path), dtype=np.uint8)
        label = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if label is None:
            continue

        # 提取多边形
        shapes = mask_to_shapes(label, CLASS_NAMES)

        # 读取图片（用于 imageData）
        img_raw = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
        img_b64 = base64.b64encode(
            cv2.imencode('.png', img)[1].tobytes()).decode()

        h, w = label.shape
        labelme_json = {
            'version': '6.3.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': img_path.name,
            'imageData': img_b64,
            'imageHeight': h,
            'imageWidth': w
        }

        json_path = OUT_DIR / f'{name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, ensure_ascii=False, indent=2)

    print(f'✅ 完成! JSON 保存至: {OUT_DIR}')


if __name__ == '__main__':
    main()
