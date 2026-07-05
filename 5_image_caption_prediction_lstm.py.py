# !pip install tensorflow pillow matplotlib
# !wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
# !wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
# !unzip -q Flickr8k_Dataset.zip
# !unzip -q Flickr8k_text.zip

import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Clear old session (important if rerunning in Colab)
tf.keras.backend.clear_session()

# -------------------------------
# 0. DOWNLOAD + UNZIP DATASET (only if needed)
# -------------------------------
if not os.path.exists("/content/Flickr8k.token.txt"):
    !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
    !unzip -q /content/Flickr8k_text.zip -d /content/

if not os.path.exists("/content/Flicker8k_Dataset") and not os.path.exists("/content/Flickr8k_Dataset"):
    !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    !unzip -q /content/Flickr8k_Dataset.zip -d /content/

# Auto-detect correct folder name
if os.path.exists("/content/Flicker8k_Dataset"):
    dataset_dir = "/content/Flicker8k_Dataset"
elif os.path.exists("/content/Flickr8k_Dataset"):
    dataset_dir = "/content/Flickr8k_Dataset"
else:
    raise FileNotFoundError("Image dataset folder not found in /content")

print("Using dataset folder:", dataset_dir)

# -------------------------------
# 1. LOAD & CLEAN CAPTIONS
# -------------------------------
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue

            image_id = tokens[0].split('.')[0]
            caption = ' '.join(tokens[1:])
            caption = caption.lower()
            caption = re.sub(r'[^a-zA-Z ]', '', caption)
            caption = 'startseq ' + caption + ' endseq'

            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions

captions_file = "/content/Flickr8k.token.txt"
captions = load_captions(captions_file)

print("Total images with captions:", len(captions))

# -------------------------------
# 2. FEATURE EXTRACTION (MobileNet)
# -------------------------------
# FIXED: add input_shape to remove warning
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
model_cnn = Model(inputs=base_model.input, outputs=x)

def extract_features(directory, limit=300):
    features = {}
    image_files = sorted(os.listdir(directory))[:limit]

    for idx, img_name in enumerate(image_files):
        path = os.path.join(directory, img_name)

        try:
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            feature = model_cnn.predict(img, verbose=0)
            image_id = img_name.split('.')[0]
            features[image_id] = feature.flatten()

            if (idx + 1) % 50 == 0:
                print(f"Extracted features for {idx+1}/{len(image_files)} images")

        except Exception as e:
            print(f"Skipping {img_name}: {e}")

    return features

features = extract_features(dataset_dir, limit=300)

if len(features) == 0:
    raise ValueError("No image features extracted. Check dataset folder.")

print("Feature shape:", list(features.values())[0].shape)
print("Images with extracted features:", len(features))

# -------------------------------
# 3. TOKENIZER
# -------------------------------
all_captions = []
for key in captions:
    all_captions.extend(captions[key])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

# Reverse mapping for fast lookup
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

print("Vocab Size:", vocab_size)
print("Max Caption Length:", max_length)

# -------------------------------
# 4. CREATE TRAINING DATA
# -------------------------------
X1, X2, y = [], [], []

for key, cap_list in captions.items():
    if key not in features:
        continue

    for cap in cap_list:
        seq = tokenizer.texts_to_sequences([cap])[0]

        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]

            # FIXED: use RIGHT PADDING for cuDNN LSTM compatibility
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]

            X1.append(features[key])
            X2.append(in_seq)
            y.append(out_seq)   # integer target (better than one-hot)

X1 = np.array(X1, dtype=np.float32)
X2 = np.array(X2, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print("X1 shape:", X1.shape)
print("X2 shape:", X2.shape)
print("y shape:", y.shape)

# -------------------------------
# 5. MODEL (CNN Feature + LSTM)
# -------------------------------
inputs1 = layers.Input(shape=(1024,))
fe1 = layers.Dense(256, activation='relu')(inputs1)

inputs2 = layers.Input(shape=(max_length,))
se1 = layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = layers.LSTM(256)(se1)

decoder = layers.add([fe1, se2])
decoder = layers.Dense(256, activation='relu')(decoder)
outputs = layers.Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()

# -------------------------------
# 6. TRAIN
# -------------------------------
history = model.fit(
    [X1, X2],
    y,
    epochs=5,
    batch_size=256,
    verbose=1
)

# Save model
model.save("/content/image_caption_model.h5")
print("Model saved to /content/image_caption_model.h5")

# -------------------------------
# 7. GENERATE CAPTION
# -------------------------------
def generate_caption(photo):
    in_text = 'startseq'

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]

        # FIXED: use RIGHT PADDING here too
        seq = pad_sequences([seq], maxlen=max_length, padding='post')

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = index_to_word.get(yhat)
        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text

# -------------------------------
# 8. TEST ON ONE IMAGE
# -------------------------------
image_name = "1000268201_693b08cb0e.jpg"   # change image here
img_path = os.path.join(dataset_dir, image_name)

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Test image not found: {img_path}")

img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

feature = model_cnn.predict(img_array, verbose=0)
photo = feature.reshape((1, 1024))

caption = generate_caption(photo)

# Clean caption for display
final_caption = caption.replace('startseq', '').replace('endseq', '').strip()

# -------------------------------
# 9. DISPLAY IMAGE + CAPTION
# -------------------------------
img = plt.imread(img_path)
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title(final_caption if final_caption else "No caption generated")
plt.axis('off')
plt.show()

print("Generated Caption:", final_caption)
