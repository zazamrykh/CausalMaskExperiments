# CausalMaskExperiments
Made with github.com/Dolmachi
Trained transformer without/with causal mask. Got the same quality but costs much more time and GPU memory.
As it known generative pre-trained transformer-like archictecures are training with causal mask. But here no strong restriction to train without it. Of course it will leads to much more time and memory consumption. 

We have modificated batch processing (see get_loss_without_mask method in train.py) for models without mask. Then we have trained 2 models and compared them. We have got results:

              With mask  |  Without mask
              
Loss (CE):       4.227   |    4.231

Time (min):       20     |    559

GPU usage (mb):  3147    |    22855

And we have not particularly different quality of generation.
You can find more about in experiment in .docx file (but it have been written on russian).
