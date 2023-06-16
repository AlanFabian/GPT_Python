#Alan de Jesus Fabian Garcia 
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparametros
batch_size = 32 # Cuantas secuencias independientes seran procesadas en paralelo
block_size = 8 # Cual es el maximo del largo que puede tener el contexto para las predicciones ?
max_iters = 3000 #Numero de maximas iteracciones 
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# todos los caracteres unicos que pueden ocurrir en el texto 
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Crea un mapeo para los caracteres e integrales 
stoi = { ch:i for i,ch in enumerate(chars) }#Diccionario con el nombre stoi string to index que mapea cada caracter unico ch en la lista de chars es decir asigna un numero entero unico a cada caracter
itos = { i:ch for i,ch in enumerate(chars) }#Diccionario Index to string que mapea cada indice 1 en la lista a su caracter correspondiente es decir asigna un caracter unico a cada numero entero
encode = lambda s: [stoi[c] for c in s] #Toma la cadena de texto s como entrada y la utiliza en el diccionario stoi para codificar cada cadena del caracter en su correspondiente numero entero
decode = lambda l: ''.join([itos[i] for i in l]) # es el decoder :toma una lista de integrales  y da como salida strings.

# Entrenamiento y testeo splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # El 90 % sera entrenaod y el porcentaje restante sera evaluado 
train_data = data[:n]
val_data = data[n:]

#Cargo los datos 
def get_batch(split):
    #Genera una pequeña muestra de datos con inputs x y objetivos y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#Modelo de biagrama simple 
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #Cada token lee directamente los logits del siguiente token en una tabla de busqueda 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx y los objetivos son ambos (B,T)tensor de numeros enteros 
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx es (B,T) matriz de indices en el contexto actual 
        for _ in range(max_new_tokens):
            #Obtener las predicciones 
            logits, loss = self(idx)
            #Se concentra nomas en el ultimo paso 
            logits = logits[:, -1, :] #Pasa a ser (B, C)
            #Aplica el softmax para obtener las probabilidades 
            probs = F.softmax(logits, dim=-1) # (B, C)
            #Muestreo de la distribucion 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #Apendice del ejemplo para correr en la secuencia 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Crear un optimizado para pytorch 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Cada cierto tiempo evalua la perdida en los conjuntos de tren val 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #Muestra ejemplos de datos 
    xb, yb = get_batch('train')

    #Evalua la perdida de datos 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Genera desde el modelo 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
