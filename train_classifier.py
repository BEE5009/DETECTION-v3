import pickle
from pathlib import Path

import numpy as np
from console_utils import configure_console_encoding, safe_print
configure_console_encoding()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from labels import THAI_LABELS, resolve_label_name

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data.pickle'
MODEL_PATH = BASE_DIR / 'model.p'
WORD_DATA_PATH = BASE_DIR / 'word_data.pickle'
WORD_MODEL_PATH = BASE_DIR / 'word_model.p'


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Training data file not found: {DATA_PATH}\nPlease run create_dataset.py first.'
        )

    with DATA_PATH.open('rb') as f:
        data_dict = pickle.load(f)

    if not data_dict.get('data'):
        raise RuntimeError('No training data found. Run create_dataset.py first.')

    return data_dict


def load_word_training_data():
    if not WORD_DATA_PATH.exists():
        return None

    with WORD_DATA_PATH.open('rb') as f:
        return pickle.load(f)


def train_and_save_model(data: np.ndarray, labels: np.ndarray, output_path: Path, model_name: str):
    if len(data) == 0 or len(labels) == 0:
        safe_print(f'No {model_name} samples were found. Skip training.')
        return False

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(data) < 4:
        x_train, x_test = data, data
        y_train, y_test = labels, labels
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=0.2,
            shuffle=True,
            stratify=labels,
        )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    safe_print(f'{model_name} accuracy: {score * 100:.2f}%')

    with output_path.open('wb') as f:
        pickle.dump({'model': model}, f)

    safe_print(f'Saved {model_name} model to {output_path}')
    return True


def main():
    try:
        data_dict = load_training_data()
    except (FileNotFoundError, RuntimeError) as exc:
        safe_print(f'Skip character training: {exc}')
    else:
        data = np.asarray(data_dict['data'], dtype=float)
        labels = np.asarray([resolve_label_name(label) for label in data_dict['labels']])
        safe_print(f'Character training: {len(data)} frames')
        train_and_save_model(data, labels, MODEL_PATH, 'character')

    word_data_dict = load_word_training_data()
    if not word_data_dict or not word_data_dict.get('data'):
        safe_print('No word training data found. Add samples to train_words/<word>/ and run create_dataset.py first.')
        return

    word_data = np.asarray(word_data_dict['data'], dtype=float)
    word_labels = np.asarray([str(label) for label in word_data_dict['labels']])
    safe_print(f'Word training: {len(word_data)} frames')
    train_and_save_model(word_data, word_labels, WORD_MODEL_PATH, 'word')


if __name__ == '__main__':
    main()
