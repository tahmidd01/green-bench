# Green AI Benchmark: PyTorch vs. TensorFlow Efficiency

## Project Overview
This project investigates whether equivalent deep learning workloads in PyTorch and TensorFlow deliver lower energy consumption and estimated carbon impact while maintaining comparable predictive quality. 

The benchmark separates framework effects from differences in architecture, data, training configuration, and measurement methods by treating the software framework as the primary independent variable. This methodology supports energy-aware model development through controlled, statistically validated comparisons.

## Pilot Experimental Results (Update 1)
Our initial pilot benchmark executed a controlled Simple CNN training workload on a CPU (AMD Ryzen 5 2600) using the CIFAR-10 dataset. 

**Key Findings:**
* **Runtime:** PyTorch completed the 1-epoch training run 2.01x faster (42.86s vs 86.00s).
* **Energy Consumption:** PyTorch used 35.2% less tracked total energy (0.706 Wh vs 1.089 Wh).
* **Emissions:** PyTorch yielded 35.2% lower estimated emissions (0.488 g CO2e vs 0.753 g CO2e).
* **Power Dynamics:** PyTorch drew a higher average tracked power (59.16 W vs 45.59 W) but finished in roughly half the time, demonstrating a "race-to-idle" efficiency pattern.
* **Parameter Parity:** Both framework implementations achieved an exact parameter match of 29,194 trainable parameters with equivalent model topologies.

*Note: This pilot represents a single-run CPU comparison. It does not establish universal framework superiority. Test accuracy, GPU behavior, and repeated-run variance are subject to further testing.*

## Controlled Experimental Design
To ensure a fair comparison, the benchmark enforces strict parity across both frameworks:
* **Dataset:** CIFAR-10 (50,000 training images, 10 classes).
* **Preprocessing:** Matched [0,1] scaling and channel normalization constants.
* **Hyperparameters:** Fixed Batch Size (64), Optimizer (Adam), Learning Rate (0.001), and Epochs (1).
* **Measurement Tooling:** CodeCarbon (v3.3.0) via Windows Energy Meter Interface (EMI) for CPU tracking, estimating regional carbon intensity (ISO code BGD).

## Future Methodology & Broader Scope
As defined in the project's broader protocol, future iterations will expand upon this pilot by incorporating:
1. **Model Diversity:** Benchmarking heavier architectures including ResNet-18, MobileNetV2, EfficientNet-B0, and Vision Transformers (ViT).
2. **Operator-Level Pipelines:** Micro-benchmarking specific operations like Convolution, Batch Normalization, and Activations to identify exact execution bottlenecks.
3. **Statistical Validation:** Transitioning to paired repeated runs (at least 5 per condition) with Shapiro-Wilk testing, Bonferroni corrections, and effect-size gating.
4. **Holistic Evaluation:** Utilizing a predefined Green-Efficiency Score (GES) and Pareto-front analysis to map the trade-offs between Macro-F1 predictive performance, throughput, memory peak, and energy costs.
