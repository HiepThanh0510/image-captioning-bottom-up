import numpy as np
import matplotlib.pyplot as plt
import time


from core.rnn_layers import *
from core.captioning_solver import CaptioningSolver
from core.coco_utils import load_coco_data

from core.utils import *
from rnn_model import CaptioningRNN

data = load_coco_data(pca_features=True)

small_data = load_coco_data(max_train=5)
small_rnn_model = CaptioningRNN(
    cell_type='rnn',
    word_to_idx=data['word_to_idx'],
    input_dim=data['train_features'].shape[1],
    hidden_dim=512,
    wordvec_dim=256,
    dtype=np.float32,
)
small_rnn_solver = CaptioningSolver(
    small_rnn_model, small_data,
    update_rule='adam',
    num_epochs=50,
    batch_size=4,
    optim_config={
     'learning_rate': 5e-3,
    },
    lr_decay=0.995,
    verbose=True, print_every=10,
)

start_time = time.time()

small_rnn_solver.train()
training_time = time.time() - start_time

# Plot the training losses
# plt.plot(small_lstm_solver.loss_history)
plt.plot(small_rnn_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()