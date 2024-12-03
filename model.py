import torch
import torch.nn as nn
from transformers import GPT2Model

class AudioGPT2(nn.Module):
    def __init__(self, num_quantizers=8, codebook_size=1024, hidden_size=768, num_classes=7):
        super(AudioGPT2, self).__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.vocab_size = self.codebook_size + 1
        self.total_vocab_size = self.num_quantizers * self.vocab_size

        self.embedding = nn.Embedding(self.total_vocab_size, hidden_size)
        self.class_embedding = nn.Embedding(num_classes, hidden_size)
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.output_layer = nn.Linear(hidden_size, self.total_vocab_size)

    def forward(self, input_ids=None, class_id=None):
        offset = torch.arange(self.num_quantizers, device=input_ids.device) * self.vocab_size
        offset = offset.unsqueeze(0).unsqueeze(0)
        input_ids = input_ids + offset

        embeddings = self.embedding(input_ids).sum(dim=2)
        class_embeddings = self.class_embedding(class_id).unsqueeze(1).expand(-1, embeddings.size(1), -1)
        combined_embeddings = embeddings + class_embeddings

        gpt_outputs = self.gpt2(inputs_embeds=combined_embeddings)
        logits = self.output_layer(gpt_outputs.last_hidden_state)
        return logits
