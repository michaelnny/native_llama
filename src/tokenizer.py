# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import AbstractSet, Any, Collection, Dict, Iterator, List, Literal, Optional, Sequence, TypedDict, Union, cast

import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe
from torch import Tensor

logger = getLogger(__name__)


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|reserved_special_token_0|>',
            '<|reserved_special_token_1|>',
            '<|reserved_special_token_2|>',
            '<|reserved_special_token_3|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|reserved_special_token_4|>',
            '<|eot_id|>',  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        self.eos_token: str = '<|end_of_text|>'
        self.eot_token: str = '<|eot_id|>'
        self.pad_token: str = '<|reserved_special_token_0|>'
        self.verifier_token: str = '<|reserved_special_token_1|>'
        self.value_token: str = '<|reserved_special_token_2|>'
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens['<|begin_of_text|>']
        self.eos_id: int = self.special_tokens[self.eos_token]
        self.eot_id: int = self.special_tokens[self.eot_token]
        self.pad_id: int = self.special_tokens[self.pad_token]
        self.verifier_id: int = self.special_tokens[self.verifier_token]  # 128003
        self.value_id: int = self.special_tokens[self.value_token]  # 128004
        self.stop_tokens = {
            self.special_tokens['<|end_of_text|>'],
            self.special_tokens['<|eot_id|>'],
        }
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal['all'], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal['all'], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def prepare_model_inputs(self, dialogs: List[List[Dict]], device: Optional[str] = 'cuda') -> Dict[str, Tensor]:
        """
        Encodes a batch of dialogs into padded token tensors and creates corresponding attention masks.

        Args:
            dialogs (List[List[Dict]]): A batch of messages objects to encode.
            device (Optional[str], optional): The device to which tensors will be moved. Defaults to 'cuda'.

        Returns:
            Dict[str, Tensor]: A dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        # Apply chat template to each dialog and convert to tensor
        encoded_dialogs = [torch.LongTensor(self.apply_chat_template(dialog)) for dialog in dialogs]

        batch_size = len(dialogs)
        max_seq_len = max([len(d) for d in encoded_dialogs])

        input_ids = torch.full((batch_size, max_seq_len), self.pad_id, dtype=torch.long)
        for i, toks in enumerate(encoded_dialogs):
            seq_len = len(toks)
            input_ids[i, max_seq_len - seq_len :] = toks  # Put tokens at the end of the row (for left-side padding)

        if device is not None:
            input_ids = input_ids.to(device)

        # Create padding mask where non-pad tokens are True
        attention_mask = (input_ids != self.pad_id).bool().to(input_ids.device)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def decode_generations(self, generation_tokens: torch.Tensor, exclude_eot: bool = False) -> List[Dict[str, Any]]:
        """
        Decodes a batch of generated token sequences into readable text, excluding pad tokens.

        Args:
            generation_tokens (torch.Tensor): A tensor containing generated token sequences.
            exclude_eot (bool): Cut the end-of-turn token.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with decoded text and corresponding token IDs.
        """
        decoded_results = []

        # Create a mask where pad tokens are False and others are True
        mask = (generation_tokens != self.eos_id) & (generation_tokens != self.pad_id)

        # Iterate over each sequence and its corresponding mask
        for token_sequence, token_mask in zip(generation_tokens, mask):
            # Use tensor masking to filter out pad tokens efficiently
            filtered_tokens = token_sequence[token_mask].tolist()

            if exclude_eot and filtered_tokens[-1] == self.eot_id:
                filtered_tokens = filtered_tokens[:-1]

            # Decode the filtered token sequence
            decoded_text = self.decode(filtered_tokens)

            # Append the result to the list
            decoded_results.append({'generation': decoded_text, 'token_ids': filtered_tokens})

        return decoded_results

    def apply_chat_template(self, message: List[Dict], text_start: bool = True, add_assistant_start: bool = True) -> List[int]:
        """
        Applies a chat template to a dialog and returns a list of token IDs.

        Args:
            dialog (List[Dict]): The dialog object to format.
            text_start (bool): Add start of text token.
            add_assistant_start (bool): Add assistant turn start prefix.

        Returns:
            List[int]: A list of token IDs representing the formatted dialog.
        """

        tokens = [self.special_tokens['<|begin_of_text|>']] if text_start else []

        for item in message:
            tokens.extend(self.encode_message(item))

        if add_assistant_start:
            # Append the start of an assistant message for model completion
            tokens.extend(self.encode_header({'role': 'assistant', 'content': ''}))

        return tokens

    def encode_message(self, item: Dict) -> List[int]:
        """
        Encodes a single message, including its header and content.

        Args:
            item (Dict): A dictionary containing 'role' and 'content' of the message.

        Returns:
            List[int]: A list of token IDs representing the encoded message.
        """
        tokens = self.encode_header(item)
        tokens.extend(self.encode(item['content'], bos=False, eos=False))
        tokens.append(self.special_tokens['<|eot_id|>'])
        return tokens

    def encode_header(self, item: Dict) -> List[int]:
        """
        Encodes the header of a item, including role information.

        Args:
            item (Dict): A dictionary containing 'role' and 'content' of the message.

        Returns:
            List[int]: A list of token IDs representing the encoded header.
        """
        tokens = [self.special_tokens['<|start_header_id|>']]
        tokens.extend(self.encode(item['role'], bos=False, eos=False))
        tokens.append(self.special_tokens['<|end_header_id|>'])
        tokens.extend(self.encode('\n\n', bos=False, eos=False))
        return tokens

    def encode_user_turn(self, content: str, add_text_start: bool = True, add_assistant_start: bool = True) -> List[int]:
        """
        Encodes the user's content, including the start part for the assistant turn's token.

        Args:
            content (str): A content of the message.
            add_text_start (bool): Whether to include the bos token.
            add_assistant_start (bool): Whether to include the assistant's start.

        Returns:
            List[int]: A list of token IDs representing the encoded assistant content.
        """
        tokens = [self.special_tokens['<|begin_of_text|>']] if add_text_start else []

        tokens.extend(self.encode_message({'role': 'user', 'content': content}))

        if add_assistant_start:
            # Append the start of an assistant message for model completion
            tokens.extend(self.encode_header({'role': 'assistant', 'content': ''}))

        return tokens

    def encode_assistant_turn(self, content: str, add_head: bool = False, eot: bool = True) -> List[int]:
        """
        Encodes the assistant's content, including the end-of-turn token.

        Args:
            content (str): A content of the message.
            add_head (bool): Whether to include the assistant's header.
            eot (bool): Whether to include the end-of-turn token.

        Returns:
            List[int]: A list of token IDs representing the encoded assistant content.
        """
        tokens = []
        if add_head:
            tokens = self.encode_header({'role': 'assistant', 'content': ''})

        tokens.extend(self.encode(content, bos=False, eos=False))
        if eot:
            tokens.append(self.special_tokens['<|eot_id|>'])
        return tokens
