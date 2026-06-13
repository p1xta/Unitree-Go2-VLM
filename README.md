# Unitree Go2 VLM System

## Overview

This repository contains a system for controlling the Unitree Go2 robot using natural voice input. The system converts speech into robot actions using a multimodal pipeline that combines speech processing, vision input, and a Vision-Language Model Qwen3-VL.

The core idea is a closed feedback loop where the robot continuously grounds its decisions in real-world perception.

---

## System Pipeline

1. The user provides a voice command.
2. Speech is converted into text via a speech processing pipeline (VAD + Keyword Spotting + STT).
3. The text command is passed to a VLM-based agent together with:
   - the current instruction
   - the latest image from the robot camera
   - previous reasoning context (actions and model outputs)
4. The VLM produces a structured JSON output:
   - either a user-facing response
   - or a robot action
5. If an action is returned:
   - the system executes it on the robot
   - after execution, a new image from the camera is captured
   - the updated state is fed back into the next VLM iteration
6. This loop continues until the task is completed or maximum amount of iterations is reached.

---

## Key Behavior

The system operates as an iterative multimodal feedback loop:

- each step depends on live camera input
- each decision depends on previous model outputs and executed actions
- the VLM is used as the central reasoning component
- execution is grounded in real-world perception
