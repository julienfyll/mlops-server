import logging
import threading
import time
from typing import Iterator, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

logger = logging.getLogger("mlops_server.engine")


class StopEventCriteria(StoppingCriteria):
    """Critère d'arrêt PyTorch qui interrompt la boucle de génération si un événement threading est activé."""

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_event.is_set()


class LLMEngine:
    def __init__(self) -> None:
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.model_name: Optional[str] = None
        self.device: str = self._detect_device()
        self.dtype: torch.dtype = self._select_dtype()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _select_dtype(self) -> torch.dtype:
        # float16 pour MPS (Mac M5) et CUDA (PC AMD ROCm), float32 pour fallback CPU
        if self.device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    def load(self, model_name: str) -> None:
        """Charge le tokenizer et les poids du modèle en mémoire."""
        logger.info(f"Chargement du modèle '{model_name}' sur {self.device} ({self.dtype})...")
        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name

        elapsed = round((time.perf_counter() - start_time), 2)
        logger.info(f"Modèle '{model_name}' prêt en {elapsed}s !")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, float]:
        """Exécute l'inférence sur le modèle et renvoie la réponse ainsi que la latence en ms."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Le modèle n'est pas encore chargé en mémoire.")

        # Application du chat template standardisé de la famille Qwen / Instruct
        messages = [
            {"role": "system", "content": "Tu es un assistant IA d'analyse et d'arbitrage logique et précis."},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        input_length = model_inputs.input_ids.shape[1]

        start_time = time.perf_counter()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Décodage uniquement des nouveaux tokens générés (sans le prompt d'origine)
        new_tokens = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return response_text, round(latency_ms, 2)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Génère du texte au fil de l'eau via un TextIteratorStreamer dans un thread d'arrière-plan."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Le modèle n'est pas encore chargé en mémoire.")

        messages = [
            {"role": "system", "content": "Tu es un assistant IA d'analyse et d'arbitrage logique et précis."},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        stopping_criteria = StoppingCriteriaList()
        if stop_event is not None:
            stopping_criteria.append(StopEventCriteria(stop_event))

        generate_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        # Exécution de model.generate dans un thread dédié pour alimenter la queue du streamer
        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        try:
            for new_text in streamer:
                if stop_event is not None and stop_event.is_set():
                    logger.info("Arrêt d'urgence du streaming demandé (client déconnecté).")
                    break
                yield new_text
        finally:
            thread.join()

    def unload(self) -> None:
        """Libère la mémoire si nécessaire."""
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

llm_engine = LLMEngine()
