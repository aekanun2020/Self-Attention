import torch
import torch.nn as nn
import torch.optim as optim

class AttentionWithGradient(nn.Module):
    def __init__(self, vocab_size, embedding_dim=4):
        super(AttentionWithGradient, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        with torch.no_grad():
            embed_weights = self.embedding.weight
            for i in range(vocab_size):
                embed_weights[i] = torch.randn_like(embed_weights[i]) * 0.1

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False) ### หมายเลข 1
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False) ### หมายเลข 2
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False) ### หมายเลข 3
        
        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
        
    def forward(self, src, target_word_idx):
        embedded = self.embedding(src)
        
        Q = self.W_q(embedded)
        K = self.W_k(embedded)
        V = self.W_v(embedded)
        
        target_Q = Q[:, target_word_idx:target_word_idx+1]
        
        # (Q * K^T) / sqrt(d_k)
        attention_scores = torch.matmul(target_Q, K.transpose(-2, -1)) / self.scale ### หมายเลข 4
        
        # softmax
        attention_weights = torch.softmax(attention_scores, dim=-1) ### หมายเลข 5
        
        # attention * V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights.squeeze()

class SizeLoss(nn.Module):
    def __init__(self, word2idx):
        super(SizeLoss, self).__init__()
        self.word2idx = word2idx
        self.size_related_words = {'elephant', 'car', 'too'}
        
    def forward(self, attention, src_tokens):
        loss = 0.0
        
        size_related_weights = []
        other_weights = []
        
        for i, token_idx in enumerate(src_tokens[0]):  # Use first batch only
            token = self.word2idx[token_idx.item()]
            weight = attention[i]
            
            if token in self.size_related_words:
                size_related_weights.append(weight)
            else:
                other_weights.append(weight)
        
        if size_related_weights:
            size_related_weights = torch.stack(size_related_weights)
            other_weights = torch.stack(other_weights)
            
            size_loss = torch.mean(1.0 - size_related_weights)
            other_loss = torch.mean(other_weights)
            
            loss = size_loss + other_loss
            
        return loss

class AttentionTrainer:
    def __init__(self, sentence, embedding_dim=4):
        self.vocab = list(set(sentence.lower().split()))
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.model = AttentionWithGradient(self.vocab_size, embedding_dim)
        self.criterion = SizeLoss(self.idx2word)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.sentence_tensor = self.sentence_to_tensor(sentence)
        self.target_word_idx = sentence.lower().split().index("too")
        
    def sentence_to_tensor(self, sentence):
        indices = [self.word2idx[word.lower()] for word in sentence.split()]
        return torch.LongTensor(indices).unsqueeze(0)
        
    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        
        output, attention = self.model(self.sentence_tensor, self.target_word_idx)
        loss = self.criterion(attention, self.sentence_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), attention
        
    def train(self, epochs=200):
        print("Starting training...")
        for epoch in range(epochs):
            loss, attention = self.train_step()
            if epoch % 20 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')
                self.analyze_attention(attention)
    
    def analyze_attention(self, attention):
        words = self.sentence_tensor.squeeze(0).tolist()
        
        print("\nAttention Pattern Analysis:")
        print("Token Importance Ranking:")
        
        word_attention = [(self.idx2word[idx], weight.item()) 
                         for idx, weight in zip(words, attention)]
        word_attention.sort(key=lambda x: x[1], reverse=True)
        
        for word, weight in word_attention:
            print(f"Token: {word:15} Attention Weight: {weight:.4f}")
        
        size_related = sum(weight for word, weight in word_attention 
                         if word in ['elephant', 'car', 'too'])
        print(f"\nCombined weight for size-related words: {size_related:.4f}")

def main():
    sentence = "The pink elephant tried to get into the car but it was too"
    trainer = AttentionTrainer(sentence, embedding_dim=4)
    trainer.train(epochs=200)

if __name__ == "__main__":
    main()