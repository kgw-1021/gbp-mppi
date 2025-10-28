#  gbp-mppi

**Gaussian Belief Propagation and Model Predictive Path Integral Control United Framework for Multi-Robot Trajectory Optimization**

---

##  Overview

`gbp-mppi`는 **Gaussian Belief Propagation (GBP)** 과 **Model Predictive Path Integral Control (MPPI)** 을 결합한  
**멀티 로봇 경로 최적화 프레임워크**입니다.  

GBP의 **분산 추론(distributed inference)** 능력을 활용해 각 로봇 간의 상호작용을 효율적으로 모델링하고,  
MPPI의 **확률적 샘플링 기반 제어(stochastic control)** 을 통해 **단일 모드 추론의 한계**를 극복합니다.  

이 프레임워크는 **협소한 환경(narrow passage)** 에서 로봇들이 병목 현상이나 dead-end에 빠지지 않고,  
**confidence-guided trajectory optimization**을 통해 안정적인 경로 탐색을 수행하도록 설계되었습니다.

---

##  Key Features

- **Distributed Path Planning via GBP**  
  - Factor Graph 기반 메시지 전달을 통해 로컬 제어 문제를 분산적으로 해결  
  - 로봇 간 상호작용을 확률적으로 모델링  

- **Mode-Aware Control via MPPI**  
  - GBP로 추론된 분포를 기반으로 샘플링 제어 수행  
  - MPPI를 통해 단일 모드 추론의 한계를 극복하고 다중 가능성 탐색  

- **Confidence-Guided Trajectory Optimization**  
  - 각 로봇의 trajectory confidence를 이용해 샘플링 가중치 및 탐색 방향 조정  

- **Multi-Robot Coordination**  
  - 협소 환경(narrow passage)에서의 상호 회피 및 동적 계획 수행  

---

##  Environment Setup

### Requirements
- **Python 3.8+**
- **PyBullet** (for physics-based simulation)

### Installation
```bash
git clone https://github.com/your-username/gbp-mppi.git

