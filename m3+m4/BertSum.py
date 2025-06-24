import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional
import logging

class BertSum:
    def __init__(self, model_name: str = "IlyaGusev/rut5_base_sum_gazeta"):
        # Определяем устройство для выполнения модели: GPU (CUDA) если доступно, иначе CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        # Инициализируем логгер для класса
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Инициализация BertSum (устройство: {self.device})")

    @property
    def model(self):
        # Ленивая загрузка модели: загружаем только при первом обращении
        if self._model is None:
            self.logger.info(f"Загрузка модели {self.model_name}")
            try:
                self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self._model.to(self.device) # Перемещаем модель на выбранное устройство
                self.logger.info("Модель успешно загружена")
            except Exception as e:
                self.logger.error(f"Не удалось загрузить модель: {str(e)}")
                raise # Повторно выбрасываем исключение после логирования
        return self._model

    @property
    def tokenizer(self):
        # Ленивая загрузка токенизатора: загружаем только при первом обращении
        if self._tokenizer is None:
            self.logger.info(f"Загрузка токенизатора {self.model_name}")
            try:
                self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.logger.info("Токенизатор успешно загружен")
            except Exception as e:
                self.logger.error(f"Не удалось загрузить токенизатор: {str(e)}")
                raise # Повторно выбрасываем исключение после логирования
        return self._tokenizer

    def summarize_text(self, text: str, max_length: int = 600, min_length: int = 50) -> Optional[str]:
        """
        Создает связную суммаризацию для длинного текста с использованием рекурсивного подхода.

        Аргументы:
            text (str): Входной текст для суммаризации.
            max_length (int): Максимальная длина генерируемого резюме.
            min_length (int): Минимальная длина генерируемого резюме.

        Возвращает:
            Optional[str]: Сгенерированное резюме или None в случае ошибки.
        """
        try:
            # 1. Токенизация всего текста
            input_ids = self.tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
            
            # Если текст достаточно короткий (в пределах лимита токенизатора), суммируем его напрямую
            if len(input_ids) <= 512:
                self.logger.info("Текст короткий, суммируем напрямую.")
                inputs = {"input_ids": input_ids.to(self.device)}
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=5, # Используем 5 лучей для более качественного результата
                    no_repeat_ngram_size=4, # Избегаем повторения n-грамм размером 4
                    early_stopping=True # Останавливаем генерацию, как только найден лучший результат
                )
                # Декодируем сгенерированные токены обратно в текст, пропуская специальные токены
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # ЭТАП "MAP": Суммаризация перекрывающихся фрагментов (чанков)
            self.logger.info("Текст длинный, используем рекурсивную суммаризацию.")
            
            chunk_size = 500  # Размер чанка (фрагмента текста) чуть меньше максимального (512)
            overlap = 50      # Перекрытие в 50 токенов для сохранения контекста между чанками
            
            chunks = []
            # Разделяем текст на чанки с заданным перекрытием
            for i in range(0, len(input_ids), chunk_size - overlap):
                chunk = input_ids[i:i + chunk_size]
                chunks.append(chunk)

            self.logger.debug(f"Всего чанков для этапа MAP: {len(chunks)}")
            
            intermediate_summaries = []
            # Суммируем каждый чанк по отдельности
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Суммирование чанка {i + 1}/{len(chunks)}")
                inputs = {"input_ids": chunk.to(self.device)}
                # Для промежуточных выжимок делаем длину поменьше, чтобы не было слишком много текста
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,  # Короче, чем финальная выжимка
                    num_beams=3,     # Меньше лучей для скорости промежуточной генерации
                    early_stopping=True
                )
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                intermediate_summaries.append(summary)

            # ЭТАП "REDUCE": Финальная суммаризация из промежуточных резюме
            combined_summary_text = " ".join(intermediate_summaries)
            self.logger.info("Генерация финального резюме из промежуточных резюме.")

            # Токенизируем объединенный текст, усекая его до максимальной длины модели
            final_input_ids = self.tokenizer.encode(combined_summary_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            final_outputs = self.model.generate(
                input_ids=final_input_ids,
                max_length=max_length,      # Используем заданную финальную длину
                min_length=min_length,
                num_beams=4,                 # Количество лучей для финальной генерации
                no_repeat_ngram_size=3,      # Чтобы избежать повторений в финальном резюме
                early_stopping=True
            )
            
            final_summary = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
            return final_summary

        except Exception as e:
            self.logger.error(f"Ошибка суммаризации: {str(e)}")
            return None