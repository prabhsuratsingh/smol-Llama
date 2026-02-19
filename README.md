# LlaMa from scratch 

Implemented Llama with Grouped Query Attention with KV Cache, Rotary Positional Embeddings (RoPE) and SentencePiece Byte-Pair Encoder from scratch using PyTorch. 

---

LlaMa Research Paper : 
[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

Grouped Query Attention Research Paper:
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

SentencePiece Byte-Pair Encoding Research Paper :
[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226)

RoPE Research Paper:
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

Hyperparameters :
```python
    vocab_size = len(tokenizer.vocab)
    num_epochs = 5
    log_interval = 100
    global_step = 0

    model = Llama(
        vocab_size=vocab_size,
        embed_size=256,
        num_layers=6,
        heads=8,
        kv=4,
    ).to(device)

    dataset = LlamaDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
```

Optimizer :
```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

Loss Function :
```python
    criterion = nn.CrossEntropyLoss()
```

Parameters :
```
Total parameters: 4,583,168
Trainable parameters: 4,583,168
```

Trained on :
```
NVIDIA GeForce GTX 1650
4GB VRAM
CUDA Version: 12.5
```