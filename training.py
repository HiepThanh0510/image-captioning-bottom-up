import numpy as np
import argparse 
import torch
import time 
import matplotlib.pyplot as plt

from core.coco_utils import load_coco_data

from core.model.rnn_model import CaptioningRNN # RNN and LSTM
from core.captioning_solver import CaptioningSolver # RNN and LSTM 
from core.model.transformer import CaptioningTransformer # transformer
from core.captioning_solver_transformer import CaptioningSolverTransformer # transformer

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


if args.model == "transformer":
    model = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30)
    
    model_solver = CaptioningSolverTransformer(model=model, 
                                               data=small_data, 
                                               idx_to_word=data['idx_to_word'],
           num_epochs=50,
           batch_size=25,
           learning_rate=0.001,
           verbose=True, print_every=10)
    start_time = time.time()
    model_solver.train()
    training_time = time.time() - start_time
    print(f"training_time = {training_time}")
    # save checkpoint
    torch.save(model.state_dict(), 'transformer_model.pt')
    print("model is saved successfully!")

    # plot the training losses.
    plt.plot(model_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

else: 
    model = CaptioningRNN(
        cell_type=args.model,
        word_to_idx=data['word_to_idx'],
        input_dim=data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256,
        dtype=np.float32)
    
    model_solver = CaptioningSolver(
        model, small_data,
        update_rule='adam',
        num_epochs=25,
        batch_size=4,
        optim_config={
        'learning_rate': 5e-3,
        },
        lr_decay=0.995,
        verbose=True, print_every=10)

    start_time = time.time()
    model_solver.train()
    training_time = time.time() - start_time
    print(f"training_time = {training_time}")
    # save checkpoint
    if args.model == "rnn":
        model.save_checkpoint(filename="rnn_checkpoint")
    else: 
        model.save_checkpoint(filename="lstm_checkpoint") 
        
    # plot the training losses
    # plt.plot(small_lstm_solver.loss_history)
    plt.plot(model_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()
