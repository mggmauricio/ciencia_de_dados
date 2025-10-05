import math
from collections import Counter
import heapq

# Texto original
text = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt, explicabo. Nemo enim ipsam voluptatem, quia voluptas sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos, qui ratione voluptatem sequi nesciunt, neque porro quisquam est, qui dolorem ipsum, quia dolor sit, amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt, ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, quam nihil molestiae consequatur, vel illum, qui dolorem eum fugiat, quo voluptas nulla pariatur?"

print("=== ANÁLISE DE ENTROPIA DO TEXTO ===\n")

# 1) Passar todos os caracteres para maiúsculas
text_upper = text.upper()
print("1) Texto em maiúsculas:")
print(text_upper[:100] + "..." if len(text_upper) > 100 else text_upper)
print(f"Tamanho: {len(text_upper)} caracteres\n")

# 2) Eliminar todos os símbolos e algarismos, mantendo apenas letras
text_letters_only = ''.join([char for char in text_upper if char.isalpha()])
print("2) Apenas letras (sem símbolos e algarismos):")
print(text_letters_only[:100] + "..." if len(text_letters_only) > 100 else text_letters_only)
print(f"Tamanho: {len(text_letters_only)} caracteres\n")

# 3) Gerar dicionário com frequências de todas as letras
letter_frequencies = Counter(text_letters_only)
total_letters = len(text_letters_only)

print("3) Frequências das letras:")
for letter in sorted(letter_frequencies.keys()):
    freq = letter_frequencies[letter]
    percentage = (freq / total_letters) * 100
    print(f"{letter}: {freq} ocorrências ({percentage:.2f}%)")
print(f"Total de letras: {total_letters}\n")

# 4) Calcular quantidade de informação de cada letra
print("4) Quantidade de informação de cada letra:")
information_content = {}
for letter, freq in letter_frequencies.items():
    probability = freq / total_letters
    info_content = -math.log2(probability)
    information_content[letter] = info_content
    print(f"{letter}: {info_content:.4f} bits")

print()

# 5) Calcular entropia do texto
entropy = 0
for letter, freq in letter_frequencies.items():
    probability = freq / total_letters
    entropy += probability * (-math.log2(probability))

print(f"5) Entropia do texto: {entropy:.4f} bits por símbolo")
print(f"Entropia máxima possível (26 letras): {math.log2(26):.4f} bits por símbolo")
print(f"Eficiência: {(entropy / math.log2(26)) * 100:.2f}%\n")

# 6) Gerar código binário otimizado (Huffman) para cada letra
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = []
    for char, freq in frequencies.items():
        heapq.heappush(heap, HuffmanNode(char=char, freq=freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_codes(root, code="", codes={}):
    if root.char is not None:
        codes[root.char] = code if code else "0"
    else:
        if root.left:
            generate_codes(root.left, code + "0", codes)
        if root.right:
            generate_codes(root.right, code + "1", codes)
    return codes

huffman_tree = build_huffman_tree(letter_frequencies)
huffman_codes = generate_codes(huffman_tree)

print("6) Códigos binários otimizados (Huffman):")
for letter in sorted(huffman_codes.keys()):
    print(f"{letter}: {huffman_codes[letter]}")

# Calcular comprimento médio do código
avg_code_length = sum(len(huffman_codes[letter]) * letter_frequencies[letter] for letter in huffman_codes) / total_letters
print(f"\nComprimento médio do código: {avg_code_length:.4f} bits por símbolo")
print(f"Compressão: {((1 - avg_code_length / 5) * 100):.2f}% (comparado com 5 bits por letra)\n")

# 7) Converter o texto para o novo código
encoded_text = ''.join([huffman_codes[letter] for letter in text_letters_only])

print("7) Texto convertido para código binário:")
print(f"Primeiros 200 bits: {encoded_text[:200]}...")
print(f"Tamanho original: {len(text_letters_only)} caracteres")
print(f"Tamanho codificado: {len(encoded_text)} bits")
print(f"Taxa de compressão: {len(encoded_text) / (len(text_letters_only) * 5):.4f}")

# Função para decodificar (demonstração)
def decode_huffman(encoded_text, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded = ""
    current_code = ""
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded += reverse_codes[current_code]
            current_code = ""
    
    return decoded

# Verificar se a decodificação funciona
decoded_text = decode_huffman(encoded_text, huffman_codes)
print(f"\nVerificação - Texto decodificado (primeiros 100 chars):")
print(decoded_text[:100])
print(f"Decodificação correta: {decoded_text == text_letters_only}")
