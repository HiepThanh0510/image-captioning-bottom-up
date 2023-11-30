# image-captioning-bottom-up

## 1. Introduction
+ Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. Most image captioning systems use an encoder-decoder framework, where an input image is encoded into an intermediate representation of the information in the image, and then decoded into a descriptive text sequence.
+ In this project, i implemented Image Captioning through 3 architectures: RNN (Recurrent Neural Network), LSTM (Long Short-Term Memory), and Transformer from scratch.

## 2. Concepts
+ __Encoder-Decoder architecture__. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.
+ __Transfer Learning__. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.
+ In terms of the encoder, I use a pretrained VGG16 to extract features from the input images into a vector of dimension 4096. After that, I reduce the dimensionality to 512 using PCA (Principal Component Analysis).
+ For the decoder, I build RNN, LSTM, and Transformer from scratch. Why do I do this? Because I want to have a detailed understanding of and insights into the architectures that have been and are making waves in AI in general and NLP in particular.
## 3. Setup

### Enviroment
I'm using PyTorch 2.0.1 and Python 3.8.16.


### Datasets
For this project, i will use the 2014 release of the [COCO dataset](https://cocodataset.org/), a standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions written by workers on Amazon Mechanical Turk. To get data:

```bash
cd core/datasets
bash get_datasets.sh
```
And below is my folder hierarchy:
```
.
├── core
│   ├── captioning_solver.py
│   ├── captioning_solver_transformer.py
│   ├── coco_utils.py
│   ├── datasets
│   │   ├── coco_captioning
│   │   ├── get_coco_captioning.sh
│   │   ├── get_datasets.sh
│   │   ├── get_imagenet_val.sh
│   │   └── imagenet_val_25.npz
│   ├── image_utils.py
│   ├── model
│   │   ├── rnn_model.py
│   │   └── transformer.py
│   ├── optim.py
│   ├── rnn_layers.py
│   └── transformer_layers.py
├── inference.py
├── lstm_checkpoint.npz
├── lstm_utils.py
├── README.md
├── rnn_checkpoint.npz
├── rnn_utils.py
├── training.py
└── transformer_model.pt
```
## 4. Usage
### Training
```python
# model options: rnn/lstm/transformer
python training.py --model "transformer" --num_samples 10000 
```
### Inference
```python
# model options: rnn/lstm/transformer
python inference.py --model "transformer" --num_samples 10000
```

## 5. Experiments

## 6. References
