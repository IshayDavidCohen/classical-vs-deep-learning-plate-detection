import random


def iou_xyxy(box_a, box_b) -> float:
    """IoU between two (xmin, ymin, xmax, ymax) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """
    Read a YOLO-format label file and convert to pixel bounding boxes.

    YOLO format per line: class x_center y_center width height (all normalised 0-1)

    Returns:
        List of (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)

            xmin = max(0, int((xc - w / 2) * img_w))
            ymin = max(0, int((yc - h / 2) * img_h))
            xmax = min(img_w, int((xc + w / 2) * img_w))
            ymax = min(img_h, int((yc + h / 2) * img_h))

            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def sample_negative_crops(
    img_h: int,
    img_w: int,
    gt_boxes: list[tuple[int, int, int, int]],
    num_negatives: int = 5,
    min_crop_size: int = 20,
    max_iou: float = 0.1,
    max_attempts: int = 200,
) -> list[tuple[int, int, int, int]]:
    """
    Randomly sample background crops that do NOT overlap with any GT plate box.

    We sample random rectangles with varied aspect ratios (roughly in the
    range of typical plate proportions) and reject any that overlap a
    ground-truth box above max_iou.

    Returns:
        List of (xmin, ymin, xmax, ymax) for accepted negative regions.
    """
    negatives = []
    attempts = 0

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1

        # Random width/height — vary size and aspect ratio
        crop_w = random.randint(min_crop_size, max(min_crop_size + 1, img_w // 3))
        crop_h = random.randint(min_crop_size, max(min_crop_size + 1, img_h // 3))

        if crop_w >= img_w or crop_h >= img_h:
            continue

        x1 = random.randint(0, img_w - crop_w)
        y1 = random.randint(0, img_h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        candidate = (x1, y1, x2, y2)

        # Reject if it overlaps any plate box
        overlaps = any(iou_xyxy(candidate, gt) > max_iou for gt in gt_boxes)
        if not overlaps:
            negatives.append(candidate)

    return negatives
