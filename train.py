import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
batch_size = 128 # How many batches to run parallel .
block_size = 256 # Input length + 1 to predict the next char.
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.2



# Start of https://www.youtube.com/watch?v=kCc8FmEb1nY

with open ('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# All the differenct chars in the input.
chars = sorted(list(set(text)))
# Count of different possible chars.
vocab_size = len(chars)


# Encode utility.
stoi = { char:index for index,char in enumerate(chars)}
# Decode utility.
itos = { index:char for index, char in enumerate(chars)}

# Encode chars to integers.
encode = lambda string: [stoi[char] for char in string]
# Decode list of integers to words.
decode = lambda list: ''.join([itos[token] for token in list ])


# Initialize a tensor .
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Initialize training data(90%) and validation data (10%).
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Defining block size for training the model in subsets of input data.
block_size = 8
train_data[:block_size+1]
print(train_data[:block_size+1])

"""
when input is tensor([18]) the target: 47
when input is tensor([18, 47]) the target: 56
when input is tensor([18, 47, 56]) the target: 57
when input is tensor([18, 47, 56, 57]) the target: 58
when input is tensor([18, 47, 56, 57, 58]) the target: 1
when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58
"""


# Batch generation.


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # x - input
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y - target
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range (eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
"""
Top(x) tensor is input while bottom(x) tensor is the representing target of each input token.

inputs:
torch.Size([4, 8])
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
targets:
torch.Size([4, 8])
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
"""
# Training stuff.
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout), )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (batch * time * channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Scores for the next chars.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

xb, yb = get_batch('train')
model = BigramLanguageModel()
m = model.to(device)


# print(logits.shape)
# print(loss) 
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # Evaluate the loss on train and val sets.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample batch of data.
    xb, yb = get_batch('train')

    # evalaute loss.
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)



with open("output.txt", "w") as f:
    f.write(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



""" 
# First generation - GPT is alive. Basic Bigram:

Ande,
Pra. d cat, win.


PTwirud y thy IBANATh lem'eg t houal
CHepomits; nyog ARURAn tes a, Ye ntant

"""

""" 
# Second generation - Self attention.

GRI'domy
Pr Awnch,
Le n hind l tho seirenes?
BUD w'done IIt touffo t haleer
LABeishanon.
KENGLourdr owome d inty nste s ncang, S:
Lerenos upacert:-w youls chr gbou d's HAhr vend:

Why bury thathou,
Wheilowilparedend meio head, and INO menctres y! tlou Prin:

COUERo y e h
ABe cied ckns thepr honck's'lat I'ssetisoth,
F y y,
BR:
CETEdind oromit?
Tampofrto in at ss th ieraknous;
F thetowshannord, s e a ouk m ghendo s
II itharand. I s, t s mfoun'se!
QUSHEO:

HEd!
BESASI aing.
ARIUNEYOPou, mas, ss ie

"""

""" 
# Third generation - Imporved softmax where each token has a query and key and communicates to previous tokens up to itself.

'Tn ouw freanganyos cdeme wulli, hee Thanusigh e se sparri ling amould flot har cuy sour sat tho o ordo 
thBh d teteelonge?
NEBSThi ndt ehery isaved prt he he.


POnd.

Ls erapere merthallll,

Filesew eafne:em a, INABU Ssord ch,
Sow auith qFurur?
Ihery tth, e tr hice ang sd sitenou , atou stins? ll a; y'nd Sere, anthwancove sint merk heras
Sodedg
ORY:
TO CHI an yoT hither,:
hed thmemect no
t,
JRUTICALAEI:
.

st theird herey, mibele, utllealy bour,
Foofo s
obre'lod  ani ivemy thin out go, kese al

"""


""" 
# Fifth generation - Multi head:

APpure know thee fear me strow letteas of thee veish'd but beten
For weldid breedy exclen it,
Einsing, this make thou canio's us croage,
The son,
Rewickled unjengen your of you behass,
I spleal, I is nam chards: of repon.

Slady: I true, my of did do son prise! this trusent;
Gtech
minghtlen in the reakowe sher;
Whyou prart that then all startemey's dur tarst a sope am the mostry to mothds had deteld of though by let ere.
Nor melie wray to evirieve Your not every gring!
Thour mots; lead,
And Yur

"""

"""
# Sixth generation with scaled up network. A basic decoder transformer.

"""