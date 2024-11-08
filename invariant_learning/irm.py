# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from networks import MNIST_CNN

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=390) #need in case using default MLP() as in original IRM paper
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=9100.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument(
    "--train_data", type=str,
    default="custom_data/train15_noise25.pt",
    help="Custom training dataset with spurious ratio and noise level")
parser.add_argument(
    "--validation_data", type=str,
    default="custom_data/val_noise25.pt",
    help="Validation dataset")
parser.add_argument(
    "--minority_env", type=str,
    default=None,
    help="Directory to minority sampling constructed from clustering")
parser.add_argument(
    "--balance_env", type=str,
    default=None,
    help="Directory to balance sampling constructed from clustering")
parser.add_argument("--random_irm", action='store_true', help="Train IRM with random data partitions")
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []

envs = [
  torch.load(flags.minority_env),
  torch.load(flags.balance_env),
  torch.load(flags.validation_data)
]

for restart in range(flags.n_restarts):
  print("Restart", restart)

  # Load MNIST, make train/val splits, and shuffle train set examples
  if flags.random_irm:
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e):
      def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
      def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
      # 2x subsample for computational convenience
      images = images.reshape((-1, 28, 28))[:, ::2, ::2]
      # Assign a binary label based on the digit; flip label with probability 0.25
      labels = (labels < 5).float()
      labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
      # Assign a color based on the label; flip the color with probability e
      colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
      # Apply the color to the image by zeroing out the other color channel
      images = torch.stack([images, images], dim=1)
      images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
      return {
      ' images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
      }
    
    envs = [
      make_environment(mnist_train[0][1::10], mnist_train[1][1::10], 0.1),
      make_environment(mnist_train[0][::10], mnist_train[1][::10], 0.5),
      make_environment(mnist_val[0], mnist_val[1], 0.7)
    ]

  # Define and instantiate the model

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        lin1 = nn.Linear(14 * 14, flags.hidden_dim) ##original 14x14
      else:
        lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out


  mlp = MLP().cuda()

  # Define loss function helpers

  def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

  def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

  def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

  # Train loop

  def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

  pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

  for step in range(flags.steps):
    for env in envs:
      logits = mlp(env['images'].cuda())
      env['nll'] = mean_nll(logits, env['labels'].cuda())
      env['acc'] = mean_accuracy(logits, env['labels'].cuda())
      env['penalty'] = penalty(logits, env['labels'].cuda())

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss = train_nll.clone()
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight 
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    test_acc = envs[2]['acc']
    if step % 100 == 0:
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_accs.append(train_acc.detach().cpu().numpy())
  final_test_accs.append(test_acc.detach().cpu().numpy())
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))
