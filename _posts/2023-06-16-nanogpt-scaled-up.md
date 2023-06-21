---
title: "(My attempt at) A Deep Learning Dev Setup: Scaling up nanogpt"
date: 2023-05-23
categories:
  - Engineering
tags:
  - Deep Learning
  - LLMs
  - GPUs
  - VSCode
toc: true
toc_label: "Contents"
---

In the past few weeks, I have been following (and greatly enjoying) [Andrej Karpathy's](https://karpathy.ai/) [Neural Networks: Zero to Hero course](https://karpathy.ai/zero-to-hero.html). Towards the [end of the final video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6050s) he "scales up" the GPT model, but (sadly for me) warns _"I would not run this on a CPU or Macbook or something like that"_. 

Having studiously worked through 12 hours of these lectures, I felt mildly deflated that I wouldn't be able reproduce this final benchmark and generate some semi-realistic Shakespeare text on my measly M1 MacBook Air. But Andrej, being the benevolent teacher that he is, left a breadcrumb in the supplementary links of the video - he mentions that he uses the [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud) which is _"the best and easiest way to spin up an on-demand GPU instance"_. 

So this short blog fills in the dots on how to create a deep learning development/research environment that uses VSCode as an IDE, [Lambda Cloud GPU instances](https://lambdalabs.com/service/gpu-cloud) for compute, as well as [Weights and Biases](https://wandb.ai/) for experiment tracking. 

**Note**: I am sure that the steps described below are completely trivial to many, however, given my initial intimidation at the prospect of doing this, and my surprise at how simple it was, I thought it would be worth writing down. **Also note** this guide was written for someone using a Mac, or Unix-like device. 
{: .notice--warning}

I cover:
1. Creation of a Lambda GPU Instance and connecting to it. 
2. Configuring VSCode for remote development, and some handy command line utilities for syncing code. 
3. Setting up Weights and Biases for experiment tracking. 

In the spirit of Andrej's lectures, I'll also point out places where I stumbled. 

## Conceptual view of development workflow
Since a picture can often clarify, below is a conceptual view of the development workflow I will describe. 
<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/nanogpt-scaled-up/ml_dev_setup.png" alt="Conceptual Overview of Dev Setup">
  <figcaption>Conceptual overview of the development setup for a deep learning project.</figcaption>
</figure>

Karpathy's lectures covered steps ***1*** and ***2*** of the development workflow shown above: interactive prototyping in Jupyter notebooks, and subsequent "cleaning up" of this code in a proper. IDE environment like VSCode. This short blogpost covers steps ***3*** onwards.


## Launching a Cloud GPU Instance
LambdaLabs has a great on-boarding and time-to-value experience (I swear I'm not being paid to say this!). Within about 10 minutes, I was able to spin up and `ssh` into a GPU instance that has most of the popular deep learning libraries pre-installed. 

The [Lambda Cloud docs](https://lambdalabs.com/blog/getting-started-with-lambda-cloud-gpu-instances), as well as the user experience is fairly self contained, however here is a brief overview of the steps you must follow:
1. Create a LambdaLabs account, which simply requires an email address (I chose an Individual account)
2. Add a payment method to LambdaLabs (credit card required)
3. Launch an Instance from the Lambda Cloud instances page. 
4. This will prompt you to create an SSH key. I created an RSA key using `ssh-keygen` from my Mac terminal, and uploaded the public key to the Lambda UI. 
5. After some time, your instance will then be available for you to `ssh` into, with a handy command line instruction that can be copied from the console (see screenshots below). 
6. *Make sure that you tear down your instance when you're done working*. Don't do what I did: get distracted, forget that your instance is up, and leave an H100 instance running overnight for no reason :man_facepalming:.


**Tip**: Before attempting to `ssh` onto your Cloud GPU instance, ensure that the ssh identity is added to your authentication agent by running `ssh-add /location/of-your/ssh-private-key`. Once you're logged in, you can view the instantaneous status of your GPU with the command `nvidia-smi`, or `watch nvidia-smi` for continuous monitoring. As we'll see later, Weights & Biases automatically tracks GPU usage statistics for you. 
{: .notice}

<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/nanogpt-scaled-up/lambdalabs_cloud_instance.png" alt="Lambda Labs Cloud Instance" >
  <figcaption>A view of the Lambda Labs cloud GPU instance page, with an H100 instance up and running</figcaption>
</figure>

## Using the Cloud GPU Instance
Now that we have the cloud instance up and running, we'd like to start getting productive. To do so, we need to accomplish the following:
1. Copy code from your `local` machine to the `remote` cloud instance
2. Edit code on the `remote` instance, preferably directly in an IDE. 
3. Launch training jobs on the remote instance. 

### Copying code to the remote
To copy code to the remote machine, we'll use the `rsync` command, which is fast and robust. In my case, I copied my entire `nanogpt` directory over to the home directory on the remote machine
```bash
% rsync -r nanogpt ubuntu@209.20.159.97:~/
```

### VSCode for remote development 
To configure remote development, we can install VSCode's [Remote Development Extension Pack](https://code.visualstudio.com/docs/remote/remote-overview):
- From the Extensions tab (Cmd-Shift-X), search for the Remote Development extension and install it. 
- Launch a remote session by opening the command palette (Cmd-Shift-P), and searching for *Remote-SSH: Connect to Host*. 
- Follow prompts to add a new host, and type in your SSH command from before (in my case, `ssh ubuntu@209.20.159.97`. You will also have to select the appropriate ssh config file, where your previously added ssh identities should be present)
- You can then navigate to your code directory from the left hand pane.  
- VSCode should prompt you to install the appropriate Python extension, which then allows you to launch jobs directly from the IDE. 

<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/nanogpt-scaled-up/vscode_remote_dev.png" alt="VSCode Remote Development" >
  <figcaption>VSCode with remote development configured - note how I am viewing code on the remote cloud gpu instance, and VSCode is using the Cloud instance's Python interpreter (Python 3.8 in this example) to run the training code. This is my reproduction of the scaled up run in <a href="https://youtu.be/kCc8FmEb1nY?t=6028">Karpathy's nanGPT lecture</a> with a final loss of 1.48</figcaption>
</figure>


## W&B for experiment tracking
A key part of "scaling up" any Machine Learning workflow is to not just using beefier compute, but also rapid *offline experimentation*: changing various hyperparameters (embedding sizes, learning rates etc.), and seeing their effects on key offline evaluation metrics (loss, precision, accuracy etc.). 

You can do this manually with Excel (yes, I've been there), or using local processes like Tensorboard, or you can hop onto the modern MLOps platform bandwagon! A thorough evaluation of which tool or platform to choose is beyond the scope of this article (see a [concise *enough* article](https://fullstackdeeplearning.com/course/2022/lecture-2-development-infrastructure-and-tooling/#7-experiment-and-model-management) here from the Full Stack Deep Learning course, or you can try to understand scary images [like this one](http://mattturck.com/wp-content/uploads/2021/12/2021-MAD-Landscape-v3.pdf) from [Matt Turck](https://mattturck.com/data2021/)). 

I decided to try out Weights and Biases (W&B), for the completely biased reasons that a) I had never used it before and wanted to try it, and b) many people in my twitter bubble rave about it, c) it is particularly useful as *centralized experiment tracker* when training a model on a remote instance. 

To expand on this last point: while we are forced to copy code back and forth to the Cloud GPU, we don't want to do the same for the results of each training run. With a few lines of code, we can instead have all the key hyperparameters, metrics, and compute statistics persisted centrally (particularly useful given that Lambda Cloud's GPU instances erase your data once terminated, and their persistent storage options are immature)


For a personal project like this W&B seems fantastic! Their on-boarding is easy, and time-to-value is fast. Following [their quickstart](https://docs.wandb.ai/quickstart), I added a run initialization near the top of the script, and refactored hyperparameters into a dataclass for reusability. 

<details markdown=1><summary markdown="span"><b><em>Code Snippet: Initializing W&B Run</em></b></summary>

```python
# Hyperparameters as a dataclass for convenience
@dataclass
class NanoGPTConfig:
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: int = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_embd: int = 384
    num_heads: int = 6
    n_layer: int = 6
    dropout: float = 0.2


# Initialize the run
params = NanoGPTConfig()

# Start a new wandb in the nanogpt project run to track this script
wandb.init(
  # set the wandb project where this run will be logged
  project="nanogpt",

  # track hyperparameters and run metadata
  config=asdict(params)
)
```
</details>

<br>
I then logged the training and validation loss during the training Loop as shown below. 


<details markdown=1><summary markdown="span"><b><em>Code Snippet: Logging Metrics during Training</em></b></summary>

```python
# The training loop:
for iter in range(params.max_iters):
    # every once in a while evaluate the loss on the train and val sets
    if iter % params.eval_interval == 0 or iter == params.max_iters - 1:
        losses = estimate_loss()
        eval_dict = {"step": iter, "train_loss": losses['train'], "val_loss": losses['val']}
        wandb.log(eval_dict) # Log to W&B
        print(eval_dict)

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    # optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
</details>
<br>
With this setup, I was able to log runs both from local environments, and the cloud environments. 

<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/nanogpt-scaled-up/wandb_run_history.png" alt="Weights and Biases Run History" >
  <figcaption>Screenshot of the Weights and Biases UI showing a history of runs, with local runs of the nanogpt training script alongside the GPU runs.</figcaption>
</figure>

**Debugging W&B on Lambda Cloud Instances**: When attempting to login, and then run Weights and Biases on the Lambda Cloud GPU instance, I saw a few errors. The section below reviews how I resolved these:
{: .notice--warning}

<details markdown=1><summary markdown="span">Resolving W&B Errors on Lambda Cloud Instances</summary>

When first attempting to login to W&B from the cloud instance, I saw an error as follows:

```bash
ubuntu@209-20-159-97:~/nanogpt$ wandb login
Traceback (most recent call last):
  File "/home/ubuntu/.local/bin/wandb", line 5, in <module>
    from wandb.cli.cli import cli
  File "/home/ubuntu/.local/lib/python3.8/site-packages/wandb/cli/cli.py", line 927, in <module>
    def launch_sweep(
  File "/usr/lib/python3/dist-packages/click/core.py", line 1234, in decorator
    cmd = command(*args, **kwargs)(f)
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 115, in decorator
    cmd = _make_command(f, name, attrs, cls)
  File "/usr/lib/python3/dist-packages/click/decorators.py", line 88, in _make_command
    return cls(name=name or f.__name__.lower().replace('_', '-'),
TypeError: __init__() got an unexpected keyword argument 'no_args_is_help'
```

This turned out to be an issue with the "click" library used by W&B, and required an upgrade to the latest click version: `pip install --upgrade click`. 

Next, when trying to run the training script, I received some cryptic errors about transport failing:

```bash
wandb: ERROR Internal wandb error: file data was not synced
Problem at: /home/ubuntu/nanogpt/v2.py 29 <module>
...
wandb: ERROR transport failed
```
These turned out to [require a protobuf downgrade](https://github.com/wandb/wandb/issues/5001): running ` pip install protobuf==3.20.0` worked. 



</details>

## Conclusion, and thoughts on the Course
To conclude, I must say a few words about [Andrej Karpathy's neural networks zero-to-hero series](https://karpathy.ai/zero-to-hero.html). I can't recommend this course enough! He is a fantastic teacher - not only introducing concepts carefully and deliberately with stunning clarity, but also sharing tons of insights about his personal "workflow" as an AI researcher. 

I find these workflow details especially useful. For example, the way he moves back and forth between notebooks and IDEs when writing code, or the way he prototypes tensor operations like multiplication or broadcasting with manageable size input examples, and careful inspection or reasoning about dimensions of outputs. This mirrors and validates a style of work that I have also (organically) arrived at - using the notebook as a REPL, and building up to cleaner code in an IDE. 

So, in the spirit of sharing "workflow details," I hope I was able to show how easy it can be to get a semi-serious developer setup going for deep learning projects. Yes - this blogpost is little more than a (verbose!) compendium of links and quickstart guides. But I hope it was useful for some people in lowering the barrier to running more serious deep learning jobs. Thanks for reading!
