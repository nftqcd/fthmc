# Test on flow based sampling

- [https://colab.research.google.com/drive/1rtVc0vMJPk3QMkd-abPMc0dS61PSQD77?usp=sharing](https://colab.research.google.com/drive/1rtVc0vMJPk3QMkd-abPMc0dS61PSQD77?usp=sharing)

- [x]  Acceptance dependence on the activation
- [x]  Acceptance dependence on the number of layers

In this note, I tested flow based sampler, dependence on activation and the number of layers.

The note is executed with GPU on google colab.

# Table of contents

# Flow based?

(neural network parametrized) change of field variable which realizes a map between qft to gaussian with tractable Jacobian.

# Quote from the note

- Quote

    > ### **Telemetry**

    > We'll measure some observables and diagnostics as we go.

    > For a batch of samples $\phi_i$, the effective sample size (ESS) is defined as

    $$\mathrm{ESS}=\frac{ \left(\frac{1}{N} \sum_i p[\phi_i]/q[\phi_i] \right)^2 }{ \frac{1}{N} \sum_i \left( p[\phi_i]/q[\phi_i] \right)^2 } \in [0,1]$$

    p:target distribution.

    q:parametrized distribution (neural net).

    > where $i$ indexes the samples. This definition normalizes the ESS to live in the range $[0,1]$. The ESS provides a useful measure of model quality that doesn't require the overall normalization of $p(x)$, where larger values indicate a better effective sampling of the desired distribution and $\mathrm{ESS} = 1$ is a perfect independent draw from the desired distribution for each sample.

    > Why not use this directly to train? It's much noisier than the KL divergences, so in practice we find it's less effective as a loss function.

    > ***Caution:**** The ESS is biased towards larger values when estimated using small batches of samples. Much like measures of autocorrelation time in MCMC approaches, a sufficiently large sample size is needed to determine whether any regions of sample space are missed.

# Setup

U(1) gauge theory on 2d lattice. L = (8, 8).

```python
# Theory
L = 8
lattice_shape = (L,L)
link_shape = (2,L,L)
beta = 2.0
u1_action = U1GaugeAction(beta)

# Model
prior = MultivariateUniform(torch.zeros(link_shape), 2*np.pi*torch.ones(link_shape))

n_layers = 16
n_s_nets = 2
hidden_sizes = [8,8]
kernel_size = 3
layers = make_u1_equiv_layers(lattice_shape=lattice_shape, n_layers=n_layers, n_mixture_comps=n_s_nets,
                             hidden_sizes=hidden_sizes, kernel_size=kernel_size)
set_weights(layers)
model = {'layers': layers, 'prior': prior}

# Training
base_lr = .001
optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)
```

# Test results for activations

I tested following smooth/ReLu-related activation functions

- **leaky ReLu (pre-trained), default**

    This is given in the note as default.

    [Leaky ReLu?](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled.png)

    [model_weight(long)](https://www.notion.so/model_weight-long-a7b3f2d4f7be495995693193f2b3f623)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%201.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%201.png)

    elapsed_time:0.0006458759307861328 [sec] (just go through)

    Accept rate: 0.2420654296875

    Topological susceptibility = 1.30 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **leaky ReLu (train by myself)**

    [Model_weight(long)](https://www.notion.so/Model_weight-long-85fe21ea11074699870d3360920621f2)

    orange = Loss, blue = ESS(the effective sample size)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%202.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%202.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%203.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%203.png)

    elapsed_time:523.0786511898041 [sec] for training (same for following activation)

    Accept rate: 0.22607421875

    Topological susceptibility = 1.27 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **ReLu**

    [ReLu?](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%204.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%204.png)

    [weight](https://www.notion.so/weight-6a0d96ddac364bdeb011c12ad8376487)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%205.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%205.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%206.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%206.png)

    elapsed_time:791.3966450691223 [sec]

    Accept rate: 0.218017578125

    Topological susceptibility = 1.26 +/- 0.04
    ... vs HMC estimate = 1.23 +/- 0.02

- **ELU**

    [ELU?](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%207.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%207.png)

    [Weight](https://www.notion.so/Weight-0cd9ba3d52e04f4e8de322da521a2950)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%208.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%208.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%209.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%209.png)

    elapsed_time:535.5670342445374 [sec]

    Accept rate: 0.221923828125

    Topological susceptibility = 1.31 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **GELU**

    [GELU?](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2010.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2010.png)

    [weight](https://www.notion.so/weight-3d4d9ba445644700883ffd97c592bd46)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2011.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2011.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2012.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2012.png)

    elapsed_time:520.0887725353241 [sec]

    Accept rate: 0.19775390625

    Topological susceptibility = 1.22 +/- 0.04
    ... vs HMC estimate = 1.23 +/- 0.02

- **CELU**

    [CELU?](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html#torch.nn.CELU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2013.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2013.png)

    [model weight](https://www.notion.so/model-weight-7fbfbfd381884638ac1cabead5ea5172)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2014.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2014.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2015.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2015.png)

    elapsed_time:537.0676226615906 [sec]

    Accept rate: 0.221923828125

    Topological susceptibility = 1.31 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **SELU**

    [SELU?](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html#torch.nn.SELU)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2016.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2016.png)

    [weight](https://www.notion.so/weight-2d3bf8882f1a4adfbde78aca575f6895)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2017.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2017.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2018.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2018.png)

    elapsed_time:526.9075174331665 [sec]

    Accept rate: 0.2294921875

    Topological susceptibility = 1.26 +/- 0.04
    ... vs HMC estimate = 1.23 +/- 0.02

- **SiLU**

    [SiLU?](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU)

    $\operatorname{silu}(x)=x * \sigma(x)$, * = element wise product, $\sigma$ = sigmoid

    [Weight](https://www.notion.so/Weight-6bb539b3bb464b6b9a2efc548bcdf2b6)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2019.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2019.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2020.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2020.png)

    elapsed_time:533.738171339035 [sec]

    Accept rate: 0.209228515625

    Topological susceptibility = 1.14 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **Sigmoid**

    [Sigmoid?](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2021.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2021.png)

    [weight](https://www.notion.so/weight-5267727ee32c4af48217eac535eccc08)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2022.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2022.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2023.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2023.png)

    elapsed_time:532.601818561554 [sec]

    Accept rate: 0.1444091796875

    Topological susceptibility = 1.08 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- **Softplus**

    [Softplus?](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2024.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2024.png)

    [weight](https://www.notion.so/weight-85abe6b277c84d1497af9f01571d769d)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2025.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2025.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2026.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2026.png)

    elapsed_time:538.184149980545 [sec]

    Accept rate: 0.1475830078125

    Topological susceptibility = 1.35 +/- 0.06
    ... vs HMC estimate = 1.23 +/- 0.02

- **Tanh**

    [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2027.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2027.png)

    [Weight](https://www.notion.so/Weight-917cb4b5fbd2448f9f6ba0497bf74207)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2028.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2028.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2029.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2029.png)

    elapsed_time:533.3189136981964 [sec]

    Accept rate: 0.24853515625

    Topological susceptibility = 1.24 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

## Summary for activation test

[Summary table for various activations](https://www.notion.so/8acb78e7b186466fadea60e1f0f8b5f4)

Hyperbolic tangent achieves top acceptance but acceptance is not so high.

(Note that, quality of neural net depends on initialization so it fluctuates.)

# Test for the number of layers

[https://colab.research.google.com/drive/1MgFVsMsNfT92AEJZIIhJ-emrg2NP9-it#scrollTo=n6dLWEGkfHUb](https://colab.research.google.com/drive/1MgFVsMsNfT92AEJZIIhJ-emrg2NP9-it#scrollTo=n6dLWEGkfHUb)

- Activations: Leaky-Relu, Tanh.
- N_layers = 8, 32, 64 (16 is the default)

- Leaky-Relu, N_layers = 8

    [Weight](https://www.notion.so/Weight-b2ee57e64d8b44e0ba4d5332ad6fad3a)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2030.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2030.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2031.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2031.png)

    elapsed_time:466.84913396835327 [sec]

    Accept rate: 0.0660400390625

    Topological susceptibility = 1.16 +/- 0.07
    ... vs HMC estimate = 1.23 +/- 0.02

- Leaky-Relu, N_layers = 32

    [weight](https://www.notion.so/weight-4136982d52f5463aabcbd9e3500c55f6)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2032.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2032.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2033.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2033.png)

    elapsed_time:641.0499613285065 [sec]

    Accept rate: 0.2867431640625

    Topological susceptibility = 1.44 +/- 0.06
    ... vs HMC estimate = 1.23 +/- 0.02

- Leaky-Relu, N_layers = 64

    [weight](https://www.notion.so/weight-ef488f237333418b9c0d6336daf4542f)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2034.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2034.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2035.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2035.png)

    elapsed_time:872.9874007701874 [sec]

    Accept rate: 0.3524169921875

    Topological susceptibility = 1.23 +/- 0.03
    ... vs HMC estimate = 1.23 +/- 0.02

- Leaky-Relu, N_layers = 128 (out of memory)

    ```python
    RuntimeError: CUDA out of memory. 
    Tried to allocate 2.00 MiB (GPU 0; 14.76 GiB total capacity; 
    13.54 GiB already allocated; 3.75 MiB free; 13.83 GiB reserved in 
    total by PyTorch)
    ```

- Tanh, N_layers = 8

    [weight](https://www.notion.so/weight-8e507f4222284dde8c3029d8fbc72d5c)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2036.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2036.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2037.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2037.png)

    elapsed_time:451.2139654159546 [sec]

    Accept rate: 0.0736083984375

    Topological susceptibility = 1.36 +/- 0.07
    ... vs HMC estimate = 1.23 +/- 0.02

- Tanh, N_layers = 32

    [weight](https://www.notion.so/weight-4723f2e94e174c63a1b3825448345121)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2038.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2038.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2039.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2039.png)

    elapsed_time:619.0787079334259 [sec]

    Accept rate: 0.2867431640625

    Topological susceptibility = 1.43 +/- 0.05
    ... vs HMC estimate = 1.23 +/- 0.02

- Tanh, N_layers = 64

    [weight](https://www.notion.so/weight-d36a944a3d5f41549888b63f8d77e131)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2040.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2040.png)

    ![Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2041.png](Test%20on%20flow%20based%20sampling%209a6d78fb771d41f4bdcbba21ae0d3a16/Untitled%2041.png)

    elapsed_time:821.7481803894043 [sec]

    Accept rate: 0.283447265625

    Topological susceptibility = 1.25 +/- 0.04
    ... vs HMC estimate = 1.23 +/- 0.02

    This is not improved from #layer = 32. Due to the gradient vanishing?

- Tanh,  N_layers = 128 (out of memory)

    ```python
    RuntimeError: CUDA out of memory. 
    Tried to allocate 2.00 MiB (GPU 0; 14.76 GiB total capacity; 
    13.54 GiB already allocated; 3.75 MiB free; 13.83 GiB reserved in 
    total by PyTorch)
    ```

[Summary table, the number of layers](https://www.notion.so/a1aecd9a536b40b3b1432c0d0cb8af4b)