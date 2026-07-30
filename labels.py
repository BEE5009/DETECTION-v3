THAI_LABELS = [
    'ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'เอก', 'โท', 'ตรี', 'จัตวา', 'สระอะ', 'สระอา', 'สระอำ', 'สระอุ', 'สระอู', 'สระแอ', 'สระเอ', '◌ั', '◌็', 'ฯ', 'ฤ', '◌์', 'ๆ'
]

POSE_COUNTS_BY_LABEL = {
    'ก': 1,
    'ข': 2,
    'ค': 2,
    'ฆ': 2,
    'ง': 2,
    'จ': 2,
    'ฉ': 2,
    'ช': 3,
    'ซ': 2,
    'ฌ': 3,
    'ญ': 2,
    'ฎ': 2,
    'ฏ': 2,
    'ฐ': 2,
    'ฑ': 2,
    'ฒ': 2,
    'ณ': 2,
    'ด': 1,
    'ต': 1,
    'ถ': 2,
    'ท': 1,
    'ธ': 3,
    'น': 1,
    'บ': 1,
    'ป': 1,
    'ผ': 2,
    'ฝ': 2,
    'พ': 1,
    'ฟ': 1,
    'ภ': 2,
    'ม': 1,
    'ย': 1,
    'ร': 2,
    'ล': 1,
    'ว': 2,
    'ศ': 2,
    'ษ': 2,
    'ส': 1,
    'ห': 1,
    'ฬ': 2,
    'อ': 1,
    'ฮ': 2,
}


def get_label_reference():
    return {index: label for index, label in enumerate(THAI_LABELS)}


def describe_label(value):
    label = resolve_label_name(value)
    if isinstance(value, str):
        try:
            index = int(value)
        except ValueError:
            return str(label)
        if 0 <= index < len(THAI_LABELS):
            return f'{index}: {THAI_LABELS[index]}'
        return str(label)

    if isinstance(value, (int, float)):
        index = int(value)
        if 0 <= index < len(THAI_LABELS):
            return f'{index}: {THAI_LABELS[index]}'

    return str(label)


def get_base_label_name(value):
    if isinstance(value, str):
        if '-ท่า' in value:
            base_label = value.split('-ท่า', 1)[0]
            if base_label in THAI_LABELS:
                return base_label
        if value in THAI_LABELS:
            return value
        try:
            index = int(value)
        except ValueError:
            return value
        if 0 <= index < len(THAI_LABELS):
            return THAI_LABELS[index]
        if 1 <= index <= len(THAI_LABELS):
            return THAI_LABELS[index - 1]
        return value

    if isinstance(value, (int, float)):
        index = int(value)
        if 0 <= index < len(THAI_LABELS):
            return THAI_LABELS[index]
        if 1 <= index <= len(THAI_LABELS):
            return THAI_LABELS[index - 1]

    return str(value)


def resolve_label_name(value):
    label = get_base_label_name(value)
    if isinstance(value, str) and '-ท่า' in value:
        return value
    return label


def normalize_labels(labels):
    return [resolve_label_name(label) for label in labels]


def get_pose_count_for_label(value):
    base_label = get_base_label_name(value)
    return POSE_COUNTS_BY_LABEL.get(base_label, 1)


def get_pose_count_for_index(index):
    if 0 <= index < len(THAI_LABELS):
        return get_pose_count_for_label(THAI_LABELS[index])
    return 1


def build_training_label_name(value, pose_index=None):
    base_label = get_base_label_name(value)
    if pose_index is None:
        return base_label

    pose_count = get_pose_count_for_label(base_label)
    if pose_count <= 1:
        return base_label

    return f'{base_label}-ท่า{pose_index}'


def get_pose_label_names(value):
    base_label = get_base_label_name(value)
    pose_count = get_pose_count_for_label(base_label)
    if pose_count <= 1:
        return [base_label]
    return [f'{base_label}-ท่า{pose}' for pose in range(1, pose_count + 1)]
