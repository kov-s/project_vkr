import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import re
import os # Импортируем модуль os для работы с путями файлов
from datetime import datetime # Импортируем datetime для создания уникальных имен файлов


class FredSummarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ai-forever/FRED-T5-large"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info("Tokenizer loaded.")
        except Exception as e:
            self.logger.error(f"Tokenizer loading failed: {str(e)}")
            raise

        try:
            self.logger.info("Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.logger.info("Model loaded.")
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess(self, text: str) -> str:
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))
            
            if current_length + paragraph_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(paragraph)
            current_length += paragraph_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def summarize(
        self, 
        text: str, 
        max_length: int = 1000,
        min_length=50,
        early_stopping=True,
        num_beams: int = 3,
        length_penalty: float = 1.5,
        no_repeat_ngram_size: int = 2,
        repetition_penalty: float = 1.5,
        temperature: float = 0.5,
        output_file_path: Optional[str] = None 
    ) -> Optional[str]:
        try:
            self.logger.info(f"Original text length: {len(text)} chars")
            text = self.preprocess(text)
            
            if not text.strip():
                self.logger.warning("Empty text after preprocessing")
                return None
                
            chunks = self.chunk_text(text)
            self.logger.info(f"Text split into {len(chunks)} chunks")
            
            summaries = []
            for idx, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    continue
                    
                chunk_tokens = len(self.tokenizer.tokenize(chunk))
                self.logger.info(f"Processing chunk {idx}/{len(chunks)} ({chunk_tokens} tokens)")
                
                input_text = f"Дай очень краткий обзор: {chunk.strip()}"
                inputs = self.tokenizer(
                    input_text,
                    max_length=384,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                try:
                    with torch.no_grad():
                        summary_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            
                            max_length=max_length,
                            num_beams=num_beams,
                            length_penalty=length_penalty,
                            early_stopping=early_stopping,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            repetition_penalty=repetition_penalty,
                            temperature=temperature,
                            do_sample=True
                        )
                    
                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summaries.append(summary)
                    self.logger.info(f"Chunk {idx} summary: {summary[:100]}...")
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.logger.warning(f"OOM on chunk {idx}, reducing beam size")
                        summary_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=max_length,
                            num_beams=1,
                            early_stopping=True
                        )
                        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        summaries.append(summary)
                    else:
                        raise
            
            if not summaries:
                self.logger.warning("No summaries generated")
                return None
                
            final_summary = " ".join(summaries)
            self.logger.info(f"Final summary length: {len(final_summary)} chars, {len(self.tokenizer.tokenize(final_summary))} tokens")
           
           
            if output_file_path:
                    try:
                        # Создаем директорию, если она не существует
                        output_dir = os.path.dirname(output_file_path)
                        if output_dir and not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            self.logger.info(f"Created output directory: {output_dir}")

                        with open(output_file_path, "w", encoding="utf-8") as f:
                            f.write(final_summary)
                        self.logger.info(f"Summary saved to: {output_file_path}")
                    except Exception as file_e:
                        self.logger.error(f"Failed to save summary to file {output_file_path}: {str(file_e)}")
                
           
           
           
           
            return final_summary

        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}", exc_info=True)
            return None