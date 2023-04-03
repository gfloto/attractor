# Fokker Plank Type Generative Model
## Initial Formulation

A random variable:  
$\gamma_t : \Omega \rightarrow \mathcal{X}$ with measure theory stuff: $(\Omega,\mathcal{F},P)$   
$\gamma_t^{-1}(S) = \{w \in \Omega | \gamma(w) \in S \}, S \in \mathcal{X}$  
$p_t(x) = P(\gamma_t^{-1}(x))$  

General sde can be written as the following:  
$d\gamma_t(S) = \mu_t(\gamma_t) dt + \sigma(\gamma_t, t)dB_t$ where $B_t$ is a weiner process.  

Alternatively this can be written as a Fokker-Planck equation:    
$$\frac{\partial}{\partial t} p_t(x) = \frac{\partial}{\partial x} [ \mu_t(x)p_t(x) ] + \frac{\partial}{\partial x^2}[ D_t(x)p_t(x) ] $$

General conservation equation is:  
$\frac{\partial}{\partial t} p_t(x) = -\nabla\cdot[f_t(x)p_t(x)]$  
We are interested in the following form of $f(\cdot)$  
$$ f(y) = \int p_t(x) \frac{x-y}{\Vert x-y \Vert^2} dx$$  
By using using this equation in expectation we see  
$f(y) = \mathbb{E}_{p_t(x)} \left[ \frac{x-y}{\Vert x-y\Vert^2} \right]$

In order to ensure that batching works properly, we can include a weiner process, and pull towards the origin s.t. taking batches in the expectation result in identical terminal distributions

$$ \frac{\partial}{\partial t}p_t(x) = -\nabla \cdot \left( \mathbb{E}_{p_t(x)} \left[ \frac{x-y}{\Vert x - y\Vert^2} -x \right] \right) + \lambda \nabla^2\cdot[ D_t(x)p_t(x) ]$$  

By taking samples, we can visualize the forward process in the following manner:  

<p align="center">
    <img src="imgs/forward.gif" width="400" height="400" />
</p>
