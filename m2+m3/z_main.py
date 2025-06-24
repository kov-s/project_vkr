import sys
import logging
from parsDe import load_data
from text_preprocessor import TextPreprocessor

def setup_logging():
    """Настройка централизованного логирования"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Очистка существующих обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Файловый обработчик
    file_handler = logging.FileHandler(
        'data_pipeline.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Консольный обработчик с UTF-8
    if sys.stdout.encoding != 'UTF-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def run_pipeline(input_file='messages.csv'):
    """Запуск полного пайплайна обработки данных"""
    logger = logging.getLogger('main')
    try:
        logger.info("Starting data processing pipeline")
        
        # 1. Загрузка данных
        logger.info("Loading data...")
        df = load_data(input_file)
        
        # 2. Предобработка текста
        logger.info("Preprocessing text...")
        processor = TextPreprocessor()
        processed_df = processor.preprocess_dataframe(df)
        
        # 3. Сохранение результатов
        logger.info("Saving results...")
        processed_df.to_csv('processed_messages.csv', index=False, encoding='utf-8')
        
        logger.info("Pipeline completed successfully")
        return processed_df
        
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise

if __name__ == '__main__':
    setup_logging()
    run_pipeline()