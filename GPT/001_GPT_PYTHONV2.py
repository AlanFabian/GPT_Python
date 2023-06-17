
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 #Cuantas secuencias independientes seran procesadas en paralelo ?
block_size = 256 #Cual es el maximo del largo que puede tener el contexto para predicciones 
max_iters = 5000#Numero de maximas iteraciones 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Todos los caracteres unicos que pueden ocurrir en el texto 
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Crea un mapeo para los caracteres de valor entero 
stoi = { ch:i for i,ch in enumerate(chars) }#Diccionario con el nombre stoi string to index que mapea cada caracter unico ch en las lista de chars es decir asigna un numero entero  unico a cada caracter 
itos = { i:ch for i,ch in enumerate(chars) }#Diccionario Index to string que mapea cada indice 1 en la lista a su caracter correspondiente es decir asigna un caracter unico a cada numero entero 
encode = lambda s: [stoi[c] for c in s] ##Toma la cadena de texto s como entrada y la utiliza en el diccionario stoi para codificar cada cadena del caracter en su correspondiente numero entero 
decode = lambda l: ''.join([itos[i] for i in l]) #es el decoder :toma una lista de integrales  y da como salida strings.

#Entrenamiento y testeo splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #El 90 sera entrenado y el porcentaje restante sera evaluado 
train_data = data[:n]
val_data = data[n:]

#Cargo  los fatos 
def get_batch(split):
    #Genera una pequeña muestra 
    data = train_data if split == 'train' else val_data#asignamos la variable data segun el valor de split si es = a train  data  se establece en el train_data de lo contrario se establece en val_data es decir que seran datos de entrenamiento o de evaluacion dependiendo de la validadcion del split 
    ix = torch.randint(len(data) - block_size, (batch_size,))#Tensor con inidices aleatorio ix dentro del rango que asignamos   que es igual al tamaño de la muestra ,esto nos asegura que los indices generados no excedan el tamaño del bloque del dato 
    x = torch.stack([data[i:i+block_size] for i in ix])#Tensor X apila los bloques de datos contiguos extraidos de la variable data,estos son extraidos utilizando los indices generados aleatoriamente en el paso anterior,basicamente crea una matriz x que contiene varias secuencias de entrada para el modelo
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])#Aplica lo mismo un tensor en Y la principal diferencia es que los bloques extraidos se desplazan un paso.Esto significa que cada bloque de y es la secuencia de salida corresponbdiente al bloque de entrada x
    x, y = x.to(device), y.to(device)#Esta linea mueve los tensores a un dispositivo especifico 
    return x, y

@torch.no_grad()#No realizaremos el seguimiento de gradientes por que vamos a calcular una metricay no requerimos retropropagacion de gradientes 
def estimate_loss():
    out = {}#Diccionario vacio Out para las perdidas calculadas
    model.eval()#Pone el modelo en modo de evaluacion
    for split in ['train', 'val']:#Iteramos sobre los conjuntos de datos de entrenamiento 
        losses = torch.zeros(eval_iters)#Tensor losses con una longitud eval_iters
        for k in range(eval_iters):
            X, Y = get_batch(split)#Se obtiene un lote de  los datos de entrada y de salida esperada y utilizando la funcion get bach
            logits, loss = model(X, Y)#Se pasan los datos de la entrada x y salida esperada y utilizando la funcion get batch
            losses[k] = loss.item()#La perdida obtenida se almacena en el tensor loss en la posicion k 
        out[split] = losses.mean()
    model.train()
    return out

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
        # Tamaño de la entrada (Muestra,Time-step,Canales)i
        # Tamañp de la salida(Muestras,Time-step,head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Calcular puntuaciones de atencion afinidades 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Realizar la agregacion ponderada de los valores 
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple cabezas de auto-atencion trabajando en paralelo"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        #Inicializamos la clase  t defino los atributos principales,representando las cabezas de atencion que seran utilizadas 
        #creamos un  lista de modulos donde cada modulo es una instancia de head con el tamaño especificado,tambien creamos una capa lineal 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    #Se realizar una propagacion hacia adelante de los datos a traves de la capa de atencion multiple,se pasa la entrada x a traves de cada una de las cabezas de atencion 
    #en paralelo utilizando la comprension de la lista  y se concatena el resultado a lo largo de la ultima dimension 

class FeedFoward(nn.Module):
    """Una capa lineal simple seguida de una de no linealidad"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),#En la primera capa lineal se realizar una transformacion lineal de la entrada n_embd a una dimension de 4 veces,luego se aplica a la funncion de activacion relu para introducir una de no linealidad 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),#En la segunda capa se realizar otra transformacion lineal de 4 veces n_embd  de 4 veces a la dimension de salida n_embd
            nn.Dropout(dropout),#Se aplica la capa de dropout a la salida de la segunda capa lineal para regularizar la salida 
        )

    def forward(self, x):
        return self.net(x)#Define la propagacion hacia adelante de los datos a traves de la capa feedforward 

class Block(nn.Module):
    """Bloque transformador:Comunicacion seguida de calculo"""

    def __init__(self, n_embd, n_head):
        # n_embd:dimension de incrustacion n_head:El numero de cabezas que nos gustaria tomar en cuenta 
        super().__init__()
        head_size = n_embd // n_head#Se calcula el tamaño de cada cabeza de atencion dividiendo la dimension de los embeddings entre el numero de cabezas
        self.sa = MultiHeadAttention(n_head, head_size)#Creamos una instancia de multiheadattention llamada sa con los parametros de n_head y head_size quue represente la capa de atencion multiple  y la capa de feedforward
        self.ffwd = FeedFoward(n_embd)#Instancia de feedforward llamada ffwd para que represente la capa 
        self.ln1 = nn.LayerNorm(n_embd)#Creamos dos capas de normalizacion ln1
        self.ln2 = nn.LayerNorm(n_embd)#y ln2 para normalizar la salida despues de la capa de atencion multiple y la capa de feed forward 

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    #Define la propagacion hacia adelante de los datos a traves del bloque de transformado 

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Cada token lee directamente los logits para el siguiente token de una tabla de busqueda 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)#Capa utilizada para asignar cada token de entrada a un vector de embeddin de tamaño n_embd y el tamaño del vocabulario se define por vocab_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)#capa de embedding de posiciones utilizando la clase nn.embedding esta capa se utiliza para asignar cada posicion de los tokens de un vector de embedding de tamaño n_embd el tamaño maximo de la secuencia es dado por block_size 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #Normalizacion de la ultima capa 
        self.lm_head = nn.Linear(n_embd, vocab_size)#Crea una capa lineal que se utiliza como una cabeza de clasificacion en el modelo.Esta capa toma la salida de los bloques de transformador y la proyecta en un espacio de logits 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx y los objetivos son ambos (B, T) tensor de números enteros
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx es (B, T) matriz de índices en el contexto actual
        for _ in range(max_new_tokens):
            #recortar idx a los últimos tokens block_size
            idx_cond = idx[:, -block_size:]
            # Obtener las predicciones 
            logits, loss = self(idx_cond)
            #Centrarse solo en el ultimo paso de tiempo
            logits = logits[:, -1, :] #se convierte en (B, C)
            # Aplicamos softmax para obtener las probabilidades
            probs = F.softmax(logits, dim=-1) # (B, C)
            #Muestreo de la distribucion 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #Agregar indice muestreado a la secuencia de ejecucion
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
#Imprimir el numero de parametros en el modelo
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Creamos un optimizador de Pytorch 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # De vez en cuando evaluamos la perdida en los conjuntos del tren y val
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Muestreo de datos 
    xb, yb = get_batch('train')

    # Evaluamos la perdida 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Generar a partir del modelo 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
