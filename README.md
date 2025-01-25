# synapse

burn rust bio inspired model zoo

ltc https://arxiv.org/abs/2006.04439

cfc https://www.nature.com/articles/s42256-022-00556-7

ncp https://www.nature.com/articles/s42256-020-00237-3

s4d https://arxiv.org/abs/2209.12951


python reference:

https://github.com/mlech26l/ncps

https://github.com/facebookresearch/flow_matching


## src/cells and src/wiring:

LTC and CFC cells are computed on top of a network scaffold (a wiring)

- cells::v1 and v1::wiring

  implementation of ltc, cfc and wiring as presented in the different papers and the reference implementation
  
  see examples/composite_waves_ltc_v1.rs
  
  ```cargo run --example composite_waves_ltc_v1```

- cells::v2::ltc

    is a refactor of the reference ltc cell with the incorporation of a second equation to model K+ gatind dynamics.
  
    thought process:
  
    hodgkin huxley model has 4 equations:
  
    - 1 for membrane capacitance -> depends on ionnic currents (core ode of the original ltc cell): ```dV/dt = (1/Cm) * [gleak(Vleak - V) + Σ(wi * si * (Ei - V))]``` (eq1)
  
    - 3 for gating ionnic currents (K, Na, ) each with a different time variable (fast, medium, slow response)
            . the fast channel reachs equilibrium value nearly instantly (let's discard it)
            . many biological neurons seem to be able to recover equilibrium without inactivation gates (let's discard it)
            . leaves out the K+ gating defined as ```dn/dt = (n∞(V) - n) / (τn + ε)``` (eq2)
            thus, (eq1) becomes ```dV/dt = (1/Cm) * [gleak(Vleak - V) + Σ(wi * ni * si * (Ei - V))]```
      
    next step is to observe if this dynamical system is sufficient to learn monostable/bistable | integrator/resonator properties by analysing phase space

- wiring::v2

    refactorr wiring for simplicity and more features
  
    can create any number of layers

    fanout connections can be defined to any other layer, thus enabling the ability to create thalamus like regions in a wiring

    see examples/composite_waves_ltc_v2.rs
  
    ```cargo run --example composite_waves_ltc_v2```

    mammalian cortex connection sparsity is around 90%, and the ratio of excitatory/inhibitory synapses is around 80/20

next steps:
wiring::v3 with multiple sensory layers



- cells::v1::s4d

  implem is wip
  s4d is ltc but as a state space model (no need for reccurent connections)


## src/flow_matching:

next step, implement ltc and s4d on flow matching generation
