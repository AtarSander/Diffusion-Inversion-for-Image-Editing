# DDIM Inv. + LoRA + CFG

16/07/2026


## Assumptions

+ $c$ - prompt embedding
+ $T=100$ - number of diffusion steps
+ $t\in\{T,T-1,\dots,2,1\}$ - timestep conditionings (during denoising)
+ $\epsilon_{\theta}(x_t,t,c)$ - $\epsilon$-pred output of base diffusion model (e.g., SDXL) with **frozen** weights

+ $\phi$, $\nu$ - **trained** LoRA adapters
+ $w$ - CFG scale 

## No-CFG Loss

$$\mathcal{L}_{\phi}:=\Bigl|\Bigl| \epsilon_{\phi}(x_{t-1},t,c) - \epsilon_{\theta}(x_{t},t,c) \Bigr|\Bigr|_2^2$$

## Shared CFG Loss

$$
\epsilon_{\text{pred}}:= w\cdot\epsilon_{\phi}(x_{t-1},t,c)+(1-w)\cdot\epsilon_{\phi}(x_{t-1},t,\emptyset)
$$


$$
\epsilon_{\text{gt}}:= w\cdot\epsilon_{\theta}(x_{t},t,c)+(1-w)\cdot\epsilon_{\theta}(x_{t},t,\emptyset)
$$



$$
\mathcal{L}_{\phi}:=\Bigl|\Bigl| \epsilon_{\text{pred}} - \epsilon_{\text{gt}} \Bigr|\Bigr|_2^2
$$

## Pair CFG Loss
$$
\epsilon_{\text{pred}}:= w\cdot\epsilon_{\phi}(x_{t-1},t,c)+(1-w)\cdot\epsilon_{\nu}(x_{t-1},t,\emptyset)
$$


$$
\epsilon_{\text{gt}}:= w\cdot\epsilon_{\theta}(x_{t},t,c)+(1-w)\cdot\epsilon_{\theta}(x_{t},t,\emptyset)
$$


$$
\mathcal{L}_{\phi,\nu}:=\Bigl|\Bigl| \epsilon_{\text{pred}} - \epsilon_{\text{gt}} \Bigr|\Bigr|_2^2
$$

## Shared Branch Loss

$$\mathcal{L}_{\phi}:=\Bigl|\Bigl| \epsilon_{\phi}(x_{t-1},t,c) - \epsilon_{\theta}(x_{t},t,c) \Bigr|\Bigr|_2^2+\Bigl|\Bigl| \epsilon_{\phi}(x_{t-1},t,\emptyset) - \epsilon_{\theta}(x_{t},t,\emptyset) \Bigr|\Bigr|_2^2$$

## Pair Branch Loss

$$\mathcal{L}_{\phi,\nu}:=\Bigl|\Bigl| \epsilon_{\phi}(x_{t-1},t,c) - \epsilon_{\theta}(x_{t},t,c) \Bigr|\Bigr|_2^2+\Bigl|\Bigl| \epsilon_{\nu}(x_{t-1},t,\emptyset) - \epsilon_{\theta}(x_{t},t,\emptyset) \Bigr|\Bigr|_2^2$$

## Proposed CFG Loss

$$
\epsilon_{\text{pred}}:= \epsilon_{\phi}(x_{t-1},t,c)
$$


$$
\epsilon_{\text{gt}}:= w\cdot\epsilon_{\theta}(x_{t},t,c)+(1-w)\cdot\epsilon_{\theta}(x_{t},t,\emptyset)
$$

$$
\mathcal{L}_{\phi}:=\Bigl|\Bigl| \epsilon_{\text{pred}} - \epsilon_{\text{gt}} \Bigr|\Bigr|_2^2
$$

