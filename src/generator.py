"""Llama generator class use local Llama model to generate answers"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from src.model import Transformer
from src.tokenizer import Tokenizer

logger = logging.getLogger()


class LlamaGenerator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        device: torch.device,
    ):
        self.model = model.to(device)

        self.tokenizer = tokenizer
        self.device = device

        self.eot_id = tokenizer.eot_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def generate_completions(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        exploring_steps: int = 0,  # Number of initial steps to use exploring sampling
    ) -> List[Dict]:
        """
        Generate actions for reinforcement learning based on batch states.

        Args:
            batch_states: List of EnvState objects with role and content.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = normal, < 1.0 = more focused, > 1.0 = more random).
            top_p: Nucleus sampling probability threshold (0.95 = consider tokens comprising top 95% probability).
            top_k: Top-k sampling parameter (0 = disabled).
            exploring_steps: Number of initial steps to use exploring sampling.

        Returns:
            List[EnvAction]: Generated actions for each state in the batch.

        Raises:
            ValueError: If states format is invalid or missing required keys.
        """
        # Ensure model is in evaluation mode
        if self.model.training:
            self.model.eval()

        # Tokenize input
        inputs = self.tokenizer.prepare_model_inputs(batch_messages, self.device)
        input_token_ids = inputs["input_ids"]
        input_attn_mask = inputs["attention_mask"]

        # Generate output
        output = self.generate(
            input_ids=input_token_ids,
            attn_mask=input_attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            exploring_steps=exploring_steps,
        )

        sequence_ids = output["token_ids"]
        # sequence_logits = output['logits']

        # Extract generated sequences
        generated_ids = sequence_ids[..., input_token_ids.size(1) :]
        # generated_logits = sequence_logits[..., input_token_ids.size(1):, :]
        outputs = self.tokenizer.decode_generations(generated_ids, exclude_eot=True)

        results = []

        for i, item in enumerate(outputs):
            results.append(
                {
                    "generation": item["generation"],
                    "token_ids": item["token_ids"],
                }
            )

        return results

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        exploring_steps: int = 0,  # Number of initial steps to use exploring sampling
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text using the model with KV caching and exploring start feature.

        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            attn_mask: Attention mask of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p filtering value
            top_k: Top-k filtering value
            exploring_steps: Number of initial generation steps to use exploring sampling

        Returns:
            Dict:
                'token_ids': Generated token ids of shape (batch_size, seq_len + generated_length)
                'logits': Logits of the token ids of shape (batch_size, seq_len + generated_length)
        """
        assert 10 <= max_new_tokens <= 4096
        assert 0 <= temperature <= 1.0
        assert 0 <= top_p <= 1.0
        assert 0 <= top_k <= self.model.params.vocab_size
        assert 0 <= exploring_steps <= max_new_tokens

        batch_size, seq_len = input_ids.shape
        max_total_length = seq_len + max_new_tokens

        assert max_total_length <= self.model.params.max_seq_len, (
            f"Total sequence length {max_total_length} exceeds model's maximum "
            f"length {self.model.params.max_seq_len}"
        )

        # Move tensors to device
        input_ids = input_ids.to(self.device)
        attention_mask = attn_mask.to(self.device)

        # Initialize generation tracking
        generated_ids = input_ids.clone()
        generated_logits = torch.empty(
            (batch_size, max_new_tokens, self.model.params.vocab_size), device="cpu"
        )
        eos_reached = torch.tensor([False] * batch_size, device=self.device)
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device=self.device)

        # Initialize KV cache
        caches: Optional[List[Dict[str, torch.Tensor]]] = None

        # First forward pass with full sequence
        initial_attn_mask = self.model.create_causal_attention_mask(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=seq_len,
            batch_size=batch_size,
            device=self.device,
            cache_position=torch.arange(seq_len, device=self.device),
        )
        outputs = self.model.forward(
            tokens=input_ids,
            start_pos=0,
            attn_mask=initial_attn_mask,
            use_cache=True,
            caches=caches,
        )
        caches = outputs.caches
        curr_seq_len = seq_len

        del initial_attn_mask

        # For cached generation, extend the original attention mask
        extended_attention_mask = torch.ones(
            (batch_size, max_total_length), dtype=torch.bool, device=self.device
        )
        extended_attention_mask[:, :seq_len] = attention_mask.bool()

        # Generate tokens one by one
        for curr_len in range(max_new_tokens):
            if all(eos_reached):
                break

            # Get last token for input
            curr_token = generated_ids[:, -1:]

            # Create attention mask for the cached generation
            cache_position = torch.tensor([curr_seq_len - 1], device=self.device)
            attn_mask = self.model.create_causal_attention_mask(
                attention_mask=extended_attention_mask[:, :curr_seq_len],
                sequence_length=1,
                target_length=curr_seq_len + 1,
                batch_size=batch_size,
                device=self.device,
                cache_position=cache_position,
            )

            # Forward pass
            outputs = self.model.forward(
                tokens=curr_token,
                start_pos=curr_seq_len,
                attn_mask=attn_mask,
                use_cache=True,
                caches=caches,
            )
            caches = outputs.caches

            # Store logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            generated_logits[:, curr_len, :] = next_token_logits.clone().cpu()

            # Sample next token
            next_tokens = self.sample_next_tokens(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                exploring=(
                    curr_len < exploring_steps
                ),  # use exploring start for first N steps
            )

            # Update only unfinished sequences
            next_tokens = next_tokens.squeeze(-1)
            next_tokens = torch.where(eos_reached, self.eos_id, next_tokens)

            # Update finished sequences mask
            eos_reached |= torch.isin(next_tokens, stop_tokens)

            # Append new tokens to sequences
            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )

            curr_seq_len += 1

        # Trim generated logits to actual length
        generated_length = generated_ids.size(1) - seq_len
        generated_logits = generated_logits[:, :generated_length, :]

        del caches, attn_mask, extended_attention_mask

        return {"token_ids": generated_ids.cpu(), "logits": generated_logits.cpu()}

    def sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        exploring: bool = False,
        exploring_eps: float = 0.15,
        exploring_alpha: float = 0.03,
    ) -> torch.Tensor:
        """
        Sample next tokens with optional exploration noise.

        Args:
            logits: Logits of shape (batch_size, vocab_size)
            temperature: Decode sampling temperature
            top_k: Keep only top k tokens with highest probability (0 to disable)
            top_p: Keep the top tokens with cumulative probability >= top_p (1.0 to disable)
            exploring: Minimum number of tokens to keep per batch
            exploring_eps: Epsilon constant to weight the priors vs. Dirichlet noise
            exploring_alpha: Parameter of the Dirichlet noise distribution

        Returns:
            torch.Tensor: Filtered probability distribution

        """
        # Greedy search
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Add exploration noise if requested
        if exploring:
            probs = self.add_dirichlet_noise(
                probs=probs, eps=exploring_eps, alpha=exploring_alpha
            )

        # Apply filtering after noise
        if top_k > 0 or top_p < 1.0:
            probs = self.top_k_top_p_filtering(probs=probs, top_k=top_k, top_p=top_p)

        # Sample from the distribution
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens

    def top_k_top_p_filtering(
        self, probs: torch.Tensor, top_k: int = 0, top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Filter a probability distribution using top-k and/or top-p (nucleus) sampling.
        Args:
            probs: Probability distribution of shape (batch_size, vocab_size)
            top_k: Keep only top k tokens with highest probability (0 to disable)
            top_p: Keep the top tokens with cumulative probability >= top_p (1.0 to disable)
        Returns:
            torch.Tensor: Filtered probability distribution
        """
        assert (
            probs.dim() == 2
        ), "probs should have dimension 2 (batch_size, vocab_size)"

        # Clone the original tensor to avoid in-place modifications
        probs = probs.clone()

        # Top-k filtering
        if top_k > 0:
            # Get the k largest values and their indices
            top_k_values, _ = torch.topk(probs, top_k, dim=-1)
            # Get the minimum probability in the top-k
            min_top_k_prob = top_k_values[:, -1].unsqueeze(-1)
            # Mask probabilities below the top-k threshold
            probs = probs.masked_fill(probs < min_top_k_prob, 0.0)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create a mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the mask to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            # Apply the mask for each item in the batch
            for i in range(probs.shape[0]):
                probs[i, sorted_indices[i][sorted_indices_to_remove[i]]] = 0.0

        # Renormalize the filtered distribution
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs_sum = probs_sum.masked_fill(probs_sum == 0, 1)  # Avoid division by zero
        probs = probs / probs_sum

        return probs

    @staticmethod
    def add_dirichlet_noise(
        probs: torch.Tensor, eps: float = 0.1, alpha: float = 0.03
    ) -> torch.Tensor:
        """
        Add Dirichlet noise to a probability distribution for exploration.

        Args:
            probs (torch.Tensor): Probability distribution of shape (batch_size, vocab_size)
            eps (float): Epsilon constant to weight the priors vs. Dirichlet noise
            alpha (float): Parameter of the Dirichlet noise distribution

        Returns:
            torch.Tensor: Modified probability distribution with added noise
        """
        batch_size, vocab_size = probs.shape
        device = probs.device

        # Generate Dirichlet noise
        noise_dist = torch.distributions.Dirichlet(
            torch.full((vocab_size,), alpha, device=device)
        )
        noise = noise_dist.sample((batch_size,))

        # Interpolate between original probabilities and noise
        mixed_probs = (1 - eps) * probs + eps * noise

        return mixed_probs

    @staticmethod
    def random_sample_from_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Sample uniformly from top K candidates."""
        top_m_values, top_m_indices = torch.topk(logits, k=k, dim=-1)
        # Create uniform probabilities for top K candidates
        probs = torch.ones_like(top_m_values) / k
        # Sample from the top M indices using the uniform probabilities
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return torch.gather(top_m_indices, -1, sampled_indices)


if __name__ == "__main__":
    from src.tokenizer import Tokenizer
    from src.model import load_pretrained_model

    pertrained_path = "/home/michael/.llama/checkpoints/Llama3.2-3B-Instruct"
    tokenizer_path = "/home/michael/llama/tokenizer.model"

    model = load_pretrained_model(pertrained_path, additional_heads=False)
    tokenizer = Tokenizer(tokenizer_path)

    device = "cuda"
    dtype = torch.bfloat16

    model.to(dtype=dtype, device=device)

    generator = LlamaGenerator(model=model, device=device, tokenizer=tokenizer)

    messages = [
        [
            {
                "role": "user",
                "content": "Tonya is buying Christmas gifts for her sisters. She has 2 sisters and wants to spend the exact same amount on each. She buys her younger sister 4 dolls that cost $15 each. She plans to buy lego sets for her older sister. They cost $20 each. How many lego sets does she buy?",
            },
        ],
        [
            {
                "role": "user",
                "content": "Nadine's dog rolls around in the mud. She spends 10 minutes hosing him off outside, then shampoos him three times, which takes 15 minutes per shampoo. How long does she spend cleaning her dog total?",
            },
        ],
        [
            {
                "role": "user",
                "content": "Ezekiel hikes as a hobby. This past summer, he did a challenging three-day hike across 50 kilometers of wilderness. The first day, he covered 10 kilometers of steep mountainside. The second day was flatter and he was able to cover half the full hike distance. How many kilometers did he have to hike on the third day to finish the hike?",
            },
        ],
        [
            {"role": "user", "content": "How many days are there in a week?"},
        ],
        [
            {
                "role": "user",
                "content": "Tell me a funny joke about a dog name Bob and a cat named Joe at a bar",
            },
        ],
        [
            {
                "role": "user",
                "content": "Alan went to the market and bought 20 eggs at the price of $2 per egg. He bought 6 chickens for the price of $8 per chicken. How much money did Alan spend at the market?",
            },
        ],
        [
            {
                "role": "user",
                "content": "Do you know what is the capital cities of US, Japan, Taiwan?",
            },
        ],
    ]

    outputs = generator.generate_completions(
        batch_messages=messages,
        temperature=0.6,
        top_p=0.9,
        top_k=0,
    )

    for i, output in enumerate(outputs):
        print(f"[Sample {i}]")
        print(output["generation"])
        print("\n\n---\n\n")
