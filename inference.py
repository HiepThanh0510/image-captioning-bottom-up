import matplotlib.pyplot as plt
import numpy as np
import argparse 
import torch

from core.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from core.image_utils import image_from_url

from core.model.rnn_model import CaptioningRNN # RNN and LSTM
from core.model.transformer import CaptioningTransformer # Transformer

np.random.seed(510)
torch.manual_seed(510)


#---------------#
# parsing params#
#---------------#
parser = argparse.ArgumentParser(description="config")
parser.add_argument("--model", 
                    default= "rnn",
                    required=True)
parser.add_argument("--num_samples",
                    default=10,
                    required=True)
args = parser.parse_args()

#-----------#
# load data #
#-----------#
data = load_coco_data(pca_features=True)

#----------------#
# get small data #
#----------------#
small_data = load_coco_data(max_train=int(args.num_samples))

#--------------#
# select model #
#--------------#
if args.model == "rnn" or args.model == "lstm":
    model = CaptioningRNN(
        cell_type=args.model,
        word_to_idx=data['word_to_idx'],
        input_dim=data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256,
        dtype=np.float32)
    # load checkpoint
    if args.model == "rnn":
        model.load_checkpoint("rnn_checkpoint.npz")
    else: 
        model.load_checkpoint("lstm_checkpoint.npz")
        
elif args.model == "transformer": 
    model = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30)
    
    # load checkpoint
    checkpoint = torch.load('transformer_model.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    print("model is loaded successfully!")

#------------#
# inference  # 
#------------# 
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        img = image_from_url(url)
        # skip missing URLs.
        if img is None: continue
        plt.imshow(img)
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()