import pandas as pd
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import sys
import io

def setup_logger(name, log_file='data_loader.log'):
    """Настройка логгера с поддержкой Unicode."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Очистка существующих обработчиков, чтобы избежать дублирования
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Файловый обработчик с UTF-8
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024, # 1 MB
            backupCount=5,        # 5 резервных файлов
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Консольный обработчик с UTF-8
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Инициализируем логгер при запуске модуля
logger = setup_logger('data_loader_module')

def safe_read_csv(file_path):
    """
    Безопасное чтение CSV только в кодировке UTF-8.
    """
    # Проверяем версию pandas для 'on_bad_lines'
    pd_version_major = int(pd.__version__.split('.')[0])
    read_csv_options = {
        'dtype': {
            'message_id': 'str',
            'text': 'str',
            'media_path': 'str',
            'tags': 'str',
            'date': 'str'
        },
        # 'parse_dates' не используется, так как даты конвертируются вручную
    }

    if pd_version_major >= 2:
        read_csv_options['on_bad_lines'] = 'warn'

    try:
        # Пытаемся читать только с UTF-8
        df = pd.read_csv(file_path, encoding='utf-8', **read_csv_options)
        logger.info(f"Файл '{os.path.basename(file_path)}' успешно прочитан с кодировкой: UTF-8")
        return df
    except UnicodeDecodeError as e:
        error_msg = f"Ошибка декодирования файла '{os.path.basename(file_path)}' с кодировкой UTF-8: {e}. Убедитесь, что файл действительно в UTF-8."
        logger.critical(error_msg)
        raise ValueError(error_msg) from e
    except FileNotFoundError:
        error_msg = f"Файл не найден при попытке чтения: {file_path}"
        logger.critical(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Неизвестная ошибка при чтении файла '{os.path.basename(file_path)}': {e}."
        logger.critical(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

def convert_dates(date_series):
    """
    Конвертация дат из различных форматов, включая Unix-таймстамп.
    Обрабатывает NaN значения после преобразования.
    """
    # Сначала попробуем преобразовать в Unix-таймстамп (секунды или миллисекунды)
    try:
        numeric_dates = pd.to_numeric(date_series, errors='coerce')
        valid_numeric_dates = numeric_dates.dropna()

        if not valid_numeric_dates.empty:
            max_val = valid_numeric_dates.max()

            if max_val > 2_000_000_000: # Эвристика для миллисекунд
                logger.info("Обнаружены потенциальные Unix-таймстампы (миллисекунды).")
                converted_dates = pd.to_datetime(valid_numeric_dates, unit='ms', errors='coerce')
                result_series = date_series.copy()
                result_series[valid_numeric_dates.index] = converted_dates

                # Пробуем форматы строк для тех, что не сконвертировались числом
                non_numeric_or_failed_indices = date_series[result_series.isna()].index
                if not non_numeric_or_failed_indices.empty:
                    logger.info("Некоторые даты не конвертировались как Unix-таймстамп. Пробуем строковые форматы.")
                    result_series.loc[non_numeric_or_failed_indices] = pd.to_datetime(
                        date_series.loc[non_numeric_or_failed_indices],
                        format='mixed', errors='coerce'
                    )
                return result_series.dt.tz_localize(None)

            elif max_val > 1_000_000_000: # Эвристика для секунд
                logger.info("Обнаружены потенциальные Unix-таймстампы (секунды).")
                converted_dates = pd.to_datetime(valid_numeric_dates, unit='s', errors='coerce')
                result_series = date_series.copy()
                result_series[valid_numeric_dates.index] = converted_dates

                non_numeric_or_failed_indices = date_series[result_series.isna()].index
                if not non_numeric_or_failed_indices.empty:
                    logger.info("Некоторые даты не конвертировались как Unix-таймстамп. Пробуем строковые форматы.")
                    result_series.loc[non_numeric_or_failed_indices] = pd.to_datetime(
                        date_series.loc[non_numeric_or_failed_indices],
                        format='mixed', errors='coerce'
                    )
                return result_series.dt.tz_localize(None)

    except Exception as e:
        logger.debug(f"Ошибка при попытке конвертации дат как Unix-таймстамп: {e}")

    # Если не Unix-таймстамп или предыдущие попытки не удались, пробуем смешанный формат строк
    try:
        converted_dates = pd.to_datetime(date_series, format='mixed', errors='coerce')
        if hasattr(converted_dates.dtype, 'tz') and converted_dates.dt.tz is not None:
             converted_dates = converted_dates.dt.tz_localize(None)
        return converted_dates
    except Exception as e:
        logger.error(f"Критическая ошибка при конвертации дат: {e}")
        return pd.Series(dtype='datetime64[ns]')

def log_dataframe_info(df):
    """Безопасное логирование информации о DataFrame, включая первые 3 строки."""
    if df.empty:
        logger.info("DataFrame пуст, информация для логирования отсутствует.")
        return

    try:
        buffer = io.StringIO()
        df.info(buf=buffer, verbose=True, show_counts=True)
        logger.info("\nИнформация о DataFrame:\n%s", buffer.getvalue())

        logger.info("Пример данных (первые 3 строки):")
        for i, row in df.head(3).iterrows():
            logger.info("Строка %d:", i)
            for col in df.columns:
                val = str(row[col])
                display_val = val[:100] + '...' if len(val) > 100 else val
                logger.info("  %s: %s", col, display_val)
    except Exception as e:
        logger.error(f"Ошибка при логировании информации о DataFrame: {e}", exc_info=True)

def load_data(file_path='messages.csv'):
    """Основная функция загрузки данных из CSV."""
    logger.info("=== Начало загрузки данных из '%s' ===", os.path.basename(file_path))

    if not os.path.exists(file_path):
        error_msg = f"Файл не найден: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        df = safe_read_csv(file_path)
        logger.info("Успешно прочитано %d строк из файла '%s'", len(df), os.path.basename(file_path))

        required_columns = ['message_id', 'text', 'media_path', 'tags', 'date']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        df = df[required_columns].copy()

        logger.info("Начало преобразования дат...")
        original_na_dates = df['date'].isna().sum()
        df['date'] = convert_dates(df['date'])

        if df['date'].isna().any():
            na_count_after_conversion = df['date'].isna().sum()
            if na_count_after_conversion > original_na_dates:
                logger.warning("Не удалось преобразовать %d дат. Заполняем текущей датой.", na_count_after_conversion)
            else:
                logger.info("Все даты успешно преобразованы или уже были NaN.")
            df['date'] = df['date'].fillna(datetime.now())
        else:
            logger.info("Все даты успешно преобразованы.")

        df['text'] = df['text'].fillna('')
        df['media_path'] = df['media_path'].fillna('')
        df['tags'] = df['tags'].fillna('')

        log_dataframe_info(df)
        logger.info("=== Загрузка данных успешно завершена для '%s' ===", os.path.basename(file_path))
        return df

    except Exception as e:
        logger.critical("Критическая ошибка при загрузке данных из '%s': %s", os.path.basename(file_path), str(e), exc_info=True)
        raise

if __name__ == '__main__':
    try:
        logger.info("=== СТАРТ ПРОГРАММЫ ===")
        # Попытка загрузить данные из файла messages.csv
        df = load_data('messages.csv')
        logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ===")
        print("\nЗагруженный DataFrame:")
        print(df.head())
    except Exception as e:
        logger.critical("=== АВАРИЙНОЕ ЗАВЕРШЕНИЕ: %s ===", str(e), exc_info=True)
        # В случае критической ошибки создаем пустой DataFrame
        df = pd.DataFrame(columns=['message_id', 'text', 'media_path', 'tags', 'date'])
        logger.info("Создан пустой DataFrame для продолжения работы после критической ошибки.")