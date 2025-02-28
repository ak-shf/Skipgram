import fitz
import numpy as np

# ------------------------
# Step 1: Extract and Preprocess Text from PDF
# ------------------------
doc = fitz.open("corpus.pdf")
print(doc)

# Extract text from the document
content = ""
for page in doc:
    content += page.get_text()
    print(page.get_text())

# Tokenize the text manually (character-wise grouping)
word = []
corpus = []
for key in content:
    if key.isalnum():
        word.append(key)
    elif key in '\n\t ' and len(word) > 0:
        corpus.append(''.join(word))
        word = []

print("Corpus:", corpus)

# Remove stopwords and convert to lowercase
stopword = ['is', 'a', 'in', 'through']
corpus_without_stopword = [w.lower() for w in corpus if w not in stopword]
print("Corpus without stopwords:", corpus_without_stopword)

# ------------------------
# Step 2: Generate Skip-gram Pairs
# ------------------------
# For a given window size, generate (context_words, target_word) pairs.
def generate_skipgram(text, window_size):
    skipgram = []
    for i, target in enumerate(text):
        # For each word, consider window_size words to the left and right as context
        context_words = text[max(0, i - window_size):i] + text[i + 1:i + window_size + 1]
        if len(context_words) == window_size * 2:
            skipgram.append((context_words, target))
    return skipgram

# For window_size = 1, each pair contains 2 context words.
skipgrams = generate_skipgram(corpus_without_stopword, 1)
print("Skipgram pairs:")
for context_words, target_word in skipgrams:
    print(f"Target: {target_word}, Context: {context_words}")

# ------------------------
# Step 3: Create One-Hot Encodings
# ------------------------
# Get the unique words from the processed corpus.
unique_words = sorted(set(corpus_without_stopword))
print("Unique words:", unique_words)  # There should be 45 unique words

def one_hot_encoding(word, unique_words):
    # Returns a list (vector) with a 1 in the position of the word and 0 elsewhere.
    return [1 if word == w else 0 for w in unique_words]

# Create a dictionary mapping each word to its one-hot vector.
one_hot_encodings = {word: one_hot_encoding(word, unique_words) for word in unique_words}

# Check that all words in the skipgram pairs exist in one_hot_encodings.

# ------------------------
# Step 4: Build Training Data for Skip-gram
# ------------------------
# For Skip-gram, we want each training example to be (target, context) for each context word.
skipgram_vector_pairs = [
    (one_hot_encodings[target_word], one_hot_encodings[word])
    for context_words, target_word in skipgrams
    for word in context_words
]

# Convert the list of pairs into NumPy arrays.
# X_train: target word one-hot vectors
# y_train: context word one-hot vectors
X_train = np.array([pair[0] for pair in skipgram_vector_pairs])
y_train = np.array([pair[1] for pair in skipgram_vector_pairs])

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ------------------------
# Step 5: Initialize Network Parameters
# ------------------------
embedding_size = 3
size_of_vocab = len(unique_words)

# Weight matrices: W1 maps from vocab to embedding; W2 maps from embedding to vocab.
w1 = np.random.uniform(-1, 1, (size_of_vocab, embedding_size))
w2 = np.random.uniform(-1, 1, (embedding_size, size_of_vocab))

learning_rate = 0.01
epochs = 1000

# Softmax function to convert logits to probabilities.
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# ------------------------
# Step 6: Define Training Function for Skip-gram
# ------------------------
def train_skipgram(X, y, W1, W2, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
             
            # Forward pass: target -> hidden -> output
            hidden_layer = np.dot(X[i], W1)  # Shape: (embedding_size,)
            output_layer = softmax(np.dot(hidden_layer, W2))  # Shape: (vocab_size,)

            # Compute cross-entropy loss for this example
            loss = -np.sum(y[i] * np.log(output_layer + 1e-9))
            total_loss += loss

            # Backpropagation: compute error and gradients
            error = output_layer - y[i]  # Shape: (vocab_size,)
            dW2 = np.outer(hidden_layer, error)  # Shape: (embedding_size, vocab_size)
            dW1 = np.outer(X[i], np.dot(W2, error))  # Shape: (vocab_size, embedding_size)

            # Update weights using gradient descent
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return W1, W2

# Train the model
W1, W2 = train_skipgram(X_train, y_train, w1, w2, learning_rate, epochs)

# ------------------------
# Step 7: Prediction Function for Skip-gram
# ------------------------
def predict_skipgram(target_word, W1, W2, one_hot_encodings, unique_words, top_n=2):
    
    target_vector = one_hot_encodings[target_word]
    hidden_layer = np.dot(target_vector, W1)
    output_layer = softmax(np.dot(hidden_layer, W2))
    
    # Get top N predicted context words
    top_indices = np.argsort(output_layer)[::-1][:top_n]
    predicted_words = [unique_words[i] for i in top_indices]
    return predicted_words

# Example Prediction: Given a target word, predict its context words.
target_word = "intelligence"
predicted_words = predict_skipgram(target_word, W1, W2, one_hot_encodings, unique_words)
print(f"Predicted context words for '{target_word}': {predicted_words}")
