import fasttext
import fasttext.model
import os
import polyglot.downloader
import polyglot.mapping
from pathlib import Path


# class Embedding(Layer):
#     def __init__(self, input_dimensions, output_dimensions, **kwargs):
#         super(Embedding, self).__init__(**kwargs)
#
#         # The amount of words in our vocabulary
#         self.vocab_size = input_dimensions
#         # When embedding a how, how many dimensions do the embedded vector have
#         self.embedding_size = output_dimensions
#
#     def build(self, input_shape):
#         self.embedding = self.add_weight("embedding", shape=[self.vocab_size, self.embedding_size])
#
#     def call(self, inputs, **kwargs):
#         return tf.nn.embedding_lookup(self.embedding, inputs)

def download_fasttext_embedding(language_code):
    home = str(Path.home())
    rel_path = f"fasttext_data/{language_code}.bin.gz"
    local_path = os.path.join(home, rel_path)

    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language_code}.300.bin.gz"
    os.system(f"wget -nc -O {local_path} {url}")

    return local_path


def load_fasttext_embedding(language_code) -> fasttext.model.WordVectorModel:
    return fasttext.load_model(download_fasttext_embedding(language_code))


def download_polyglot_embedding(language_code):
    downloader = polyglot.downloader.downloader
    downloader.download(f"embeddings2.{language_code}")
    folder = downloader.download_dir
    filename = downloader.info(f"embeddings2.{language_code}").filename
    path = os.path.join(folder, filename)

    return path


def load_polyglot_embedding(language_code):
    return polyglot.mapping.Embedding.load(download_polyglot_embedding(language_code))
