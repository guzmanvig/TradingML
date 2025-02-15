from collections import namedtuple

import torch

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    # TODO: convert batch.state and the others to tensors!
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)