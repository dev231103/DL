import os, pickle, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Dropout, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- PATHS ----------------
images_dir = "Flicker8k_Dataset" if os.path.exists("Flicker8k_Dataset") else "Flickr8k_Dataset"
captions_file = "Flickr8k.token.txt"
train_file = "Flickr_8k.trainImages.txt"
val_file = "Flickr_8k.devImages.txt"
test_file = "Flickr_8k.testImages.txt"

print("Images dir:", images_dir, os.path.exists(images_dir))

# ---------------- LOAD SPLITS ----------------
def load_set(file):
    return [line.strip() for line in open(file, "r") if line.strip()]

train_imgs = load_set(train_file)
val_imgs = load_set(val_file)
test_imgs = load_set(test_file)

# For faster runtime (can comment out for full dataset)
# train_imgs = train_imgs[:3000]
# val_imgs = val_imgs[:500]
# test_imgs = test_imgs[:200]

# ---------------- LOAD CAPTIONS ----------------
def clean(txt):
    txt = txt.lower()
    txt = ''.join(c for c in txt if c.isalpha() or c == ' ')
    txt = ' '.join(txt.split())
    return 'startseq ' + txt + ' endseq'

captions = {}
for line in open(captions_file, "r"):
    parts = line.strip().split('\t')
    if len(parts) != 2:
        continue
    img, cap = parts
    img = img.split('#')[0]
    captions.setdefault(img, []).append(clean(cap))

train_imgs = [x for x in train_imgs if x in captions]
val_imgs = [x for x in val_imgs if x in captions]
test_imgs = [x for x in test_imgs if x in captions]

# ---------------- TOKENIZER ----------------
all_train_caps = []
for img in train_imgs:
    all_train_caps.extend(captions[img])

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_train_caps)

vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(c.split()) for c in all_train_caps)
index_to_word = {i: w for w, i in tokenizer.word_index.items()}

print("Train:", len(train_imgs), "Val:", len(val_imgs), "Test:", len(test_imgs))
print("Vocab:", vocab_size, "MaxLen:", max_len)

# ---------------- INCEPTIONV3 FEATURE EXTRACTOR ----------------
base_model = InceptionV3(weights="imagenet")
cnn = Model(base_model.input, base_model.layers[-2].output)  # 2048-d features

def extract_feature(img_name):
    path = os.path.join(images_dir, img_name)
    img = load_img(path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = cnn.predict(img, verbose=0)
    return feat[0]

# ---------------- FEATURE CACHE ----------------
cache_file = "flickr8k_inceptionv3_features.pkl"
all_imgs = list(set(train_imgs + val_imgs + test_imgs))

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        features = pickle.load(f)
    print("Loaded cached features:", len(features))
else:
    features = {}
    for i, img_name in enumerate(all_imgs):
        try:
            features[img_name] = extract_feature(img_name)
        except:
            pass
        if (i + 1) % 500 == 0:
            print(f"Extracted {i+1}/{len(all_imgs)}")
    with open(cache_file, "wb") as f:
        pickle.dump(features, f)
    print("Saved features:", len(features))

train_imgs = [x for x in train_imgs if x in features]
val_imgs = [x for x in val_imgs if x in features]
test_imgs = [x for x in test_imgs if x in features]

# ---------------- GENERATOR ----------------
class CaptionGenerator(Sequence):
    def __init__(self, img_list, captions, features, tokenizer, max_len, vocab_size, batch_size=64):
        self.samples = []
        self.captions = captions
        self.features = features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        for img in img_list:
            for cap in captions[img]:
                self.samples.append((img, cap))

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        X1, X2, y = [], [], []

        for img, cap in batch:
            feat = self.features[img]
            seq = self.tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_len, padding='post')[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                X1.append(feat)
                X2.append(in_seq)
                y.append(out_seq)

        return (np.array(X1), np.array(X2)), np.array(y)

train_gen = CaptionGenerator(train_imgs, captions, features, tokenizer, max_len, vocab_size, batch_size=64)
val_gen = CaptionGenerator(val_imgs, captions, features, tokenizer, max_len, vocab_size, batch_size=64)

# ---------------- MODEL (InceptionV3 + GRU) ----------------
i1 = Input(shape=(2048,))
f1 = Dropout(0.5)(i1)
f2 = Dense(256, activation='relu')(f1)

i2 = Input(shape=(max_len,))
s1 = Embedding(vocab_size, 256, mask_zero=True)(i2)
s2 = Dropout(0.5)(s1)
s3 = GRU(256)(s2)

d1 = Add()([f2, s3])
d2 = Dense(256, activation='relu')(d1)
out = Dense(vocab_size, activation='softmax')(d2)

model = Model([i1, i2], out)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# ---------------- TRAIN ----------------
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ckpt = ModelCheckpoint("best_inception_gru_caption.keras", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    callbacks=[es, ckpt],
    verbose=1
)

# ---------------- PLOT LOSS ----------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# ---------------- BEAM SEARCH DECODING ----------------
def beam_search_caption(photo, beam_width=3):
    sequences = [(['startseq'], 0.0)]

    for _ in range(max_len):
        all_candidates = []

        for seq_words, score in sequences:
            if seq_words[-1] == 'endseq':
                all_candidates.append((seq_words, score))
                continue

            seq = tokenizer.texts_to_sequences([' '.join(seq_words)])[0]
            seq = pad_sequences([seq], maxlen=max_len, padding='post')

            preds = model.predict([np.array([photo]), seq], verbose=0)[0]
            top_idx = np.argsort(preds)[-beam_width:]

            for idx in top_idx:
                word = index_to_word.get(idx)
                if word is None:
                    continue
                candidate = (seq_words + [word], score - np.log(preds[idx] + 1e-10))
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    best_seq = sequences[0][0]
    caption = ' '.join(best_seq)
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption

# ---------------- TEST ----------------
sample = test_imgs[4]
feat = features[sample]
pred = beam_search_caption(feat, beam_width=3)

print("\nSample Image:", sample)
print("Predicted Caption:", pred)

print("\nActual Captions:")
for c in captions[sample]:
    print("-", c.replace("startseq ", "").replace(" endseq", ""))

img = Image.open(os.path.join(images_dir, sample))
plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.title("Predicted: " + pred)
plt.show()

model.save("final_inception_gru_caption_model.keras")
print("Saved: final_inception_gru_caption_model.keras")