Smart Electric Vehicle Grid Integration System with V2G Optimization
Project Overview
This project implements an advanced Vehicle-to-Grid (V2G) optimization system that manages a large-scale electric vehicle charging network integrated with the IEEE 33-Bus power distribution system. The system uses GPU acceleration to handle 25,000 EVs efficiently and implements sophisticated algorithms for both grid stability and financial optimization.

Key Components
1. Power System Integration (PSO.py)
Implements the IEEE 33-Bus Forward-Backward Sweep algorithm
Base system parameters:
Base Power: 100 MVA
Base Voltage: 12.66 kV
Handles power flow calculations and grid stability analysis
Models the complete distribution network with 33 buses
2. V2G Environment (code.py)
Implements a sophisticated V2G decision-making system using Reinforcement Learning
Key features:
Dynamic tariff management
Real-time SOC (State of Charge) monitoring
Grid demand factor integration
Flexible charging/discharging schedules
Parameters:
Charging power range: 3.0 kW to 22.0 kW
Energy per slot: 1.5 kWh to 11.0 kWh
Considers battery capacity and SOC constraints
3. Performance Visualization (new.py)
Implements four key analytical visualizations:

EV Load Analysis

Tracks charging and discharging patterns
Visualizes net EV load on the grid
Monitors peak charging and V2G discharge rates
Dynamic Tariff Structure

Color-coded rate visualization
Range analysis: ₹5-11/kWh
Real-time tariff tracking
Action Distribution

Monitors three states: CHARGE, DISCHARGE, and HOLD
Time-based analysis of EV behavior
GPU-accelerated event processing
Financial Performance

Real-time profit/loss tracking
Cumulative cost and revenue analysis
Financial optimization metrics
Technical Features
GPU Acceleration

Handles 25,000 EVs simultaneously
Optimized data processing
Real-time performance metrics
Smart Scheduling

48 time slots (30-minute intervals)
Dynamic pricing integration
Demand-response optimization
Financial Optimization

Real-time cost-benefit analysis
Dynamic tariff-based decision making
Profit maximization algorithms
Applications
Grid Management

Load balancing
Peak shaving
Voltage regulation
Financial Planning

Revenue optimization
Cost minimization
ROI analysis
Environmental Impact

Grid efficiency optimization
Renewable energy integration
Carbon footprint reduction
Innovation Highlights
Scalability: GPU acceleration enables handling of 25,000 EVs in real-time
Intelligence: Advanced RL-based decision making for V2G operations
Integration: Seamless integration with IEEE 33-Bus system
Financial Optimization: Real-time profit maximization and cost control
Visualization: Comprehensive real-time monitoring and analysis tools
This project represents a significant advancement in V2G technology, combining power systems engineering, machine learning, and financial optimization to create a sustainable and profitable EV charging ecosystem.
