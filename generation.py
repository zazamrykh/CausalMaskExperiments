import enum
import math

import torch

from datasets import encode, tokenizer, decoder
from utils import device, logger


class Strategies(enum.Enum):
    greedy = (0, None)
    top_k = (1, 5)
    beam = (2, 10)

    def __init__(self, num, param):
        self.num = num
        self.param = param


def continue_sentence(sentence, model, max_len=50, strategy=Strategies.greedy, alpha=0.7, device=device, llama=False):
    model.eval()

    tokens = list(encode(sentence.lower()))

    if strategy == Strategies.beam:
        n_beams = Strategies.beam.param
        beams = [tokens.copy() for _ in range(n_beams)]
        likelihoods = [0.0] * n_beams  # logartythms

    # TODO: Your code here
    iteration = -1
    while len(tokens) <= max_len:
        iteration += 1
        input = torch.tensor(tokens).to(device)[None,]
        outputs = model(input)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if llama:
            outputs = outputs['logits']
        distribution = torch.softmax(outputs[0][-1], dim=0).detach()

        if strategy == Strategies.greedy:
            token = torch.argmax(distribution)
            tokens.append(token)

        elif strategy == Strategies.top_k:
            k = Strategies.top_k.param
            indices = torch.topk(distribution, k).indices
            probabilities = distribution[indices]
            top_k_distribution = probabilities / torch.sum(probabilities)

            index = top_k_distribution.multinomial(num_samples=1, replacement=True)
            token = indices[index]
            tokens.append(token)

        elif strategy == Strategies.beam:
            if len(beams[0]) > max_len:
                break
            candidate_beams = []
            candidate_likelihoods = []
            for i, (beam, likelihood) in enumerate(zip(beams, likelihoods)):
                input = torch.tensor(beam).to(device)[None,]
                outputs = model(input)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if llama:
                    outputs = outputs['logits']
                distribution = torch.softmax(outputs[0][-1], dim=0).detach()

                indices = torch.topk(distribution,
                                     n_beams).indices  # actually, here we can use other parameter instead of n_beams
                probabilities = distribution[indices]

                for index, probability in zip(indices, probabilities):
                    new_beam = beam + [index.item()]
                    length_penalty = len(new_beam) ** alpha
                    new_likelihood = (likelihood + math.log(probability.item())) / length_penalty
                    candidate_beams.append(new_beam)
                    candidate_likelihoods.append(new_likelihood)

                if iteration == 0:
                    break

            best_beams_indices = sorted(range(len(candidate_likelihoods)), key=lambda sub: candidate_likelihoods[sub])[
                                 -n_beams:]
            beams = [candidate_beams[i] for i in best_beams_indices]
            likelihoods = [candidate_likelihoods[i] for i in best_beams_indices]

        else:
            logger.error("Choosen strategy is not available!")
            return

    if strategy == Strategies.beam:
        best_index, best_likelihood = -1, -1
        for i, likelihood in enumerate(likelihoods):
            if likelihood > best_likelihood:
                best_index = i
        tokens = beams[best_index]

    return decoder.decode(tokenizer.convert_ids_to_tokens(tokens))
